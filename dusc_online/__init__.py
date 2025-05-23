from pathlib import Path
import base64
from io import BytesIO

from trame.decorators import change, TrameApp
from trame.app import get_server
from trame.widgets import html, client
from trame.ui.html import DivLayout
from trame_image_tools.widgets import TrameImage, TrameImageRoi
from trame.ui.vuetify3 import SinglePageLayout
from trame.widgets import vuetify3 as vuetify

import stempy.io as stio
import numpy as np
from numba import jit, prange
from matplotlib import cm

from PIL import Image


@TrameApp()
class ExampleApp:
    def __init__(self, server=None):
        self._server = get_server(server, client_type="vue3")

        self.loaded = False
        self.sa = None
        self.dp = None
        self.rs = None
        self.fr_rows = None
        self.fr_cols = None

        self.fr_full = None
        self.fr_full_3d = None

        self.num_frames_per_scan = None

        self.scan_dimensions = (0, 0)
        self.frame_dimensions = (576, 576)

        self.state.real_space_roi = [0, 0, 10, 10]
        self.state.diffraction_space_roi = [0, 0, 10, 10]
            
        self.file_paths = {
            'Label1': 'C:/users/linol/Downloads/FOURD_250415_1407_27734_00011.h5',
            'Label2': 'C:/users/linol/Downloads/FOURD_250106_1729_24066_00005.h5',
            'Label3': 'C:/users/linol/Downloads/FOURD_250106_1730_24067_00006.h5',
            }

        self.state.dataset_names = list(self.file_paths.values())
        self.state.selected_dataset = self.state.dataset_names[1]
        
        self.setData(self.state.selected_dataset)

        self.ui = None
        self._build_ui()

    @property
    def server(self):
        return self._server

    @property
    def state(self):
        return self.server.state

    @property
    def ctrl(self):
        return self.server.controller

    @change("diffraction_space_roi")
    def update_real(self, *args, **kwargs):
        if not self.loaded:
            return

        self.rs[:] = self.getImage_jit(self.fr_rows, self.fr_cols,
            self.state.diffraction_space_roi[1] - 1,
            self.state.diffraction_space_roi[1] + self.state.diffraction_space_roi[3] + 0,
            self.state.diffraction_space_roi[0] - 1,
            self.state.diffraction_space_roi[0] + self.state.diffraction_space_roi[2] + 0,
        )

        real_c_data = self.apply_colormap(self.rs, self.scan_dimensions, cm.cividis, False)
        real_im = Image.fromarray(real_c_data)
        self.state.real_image = self.convert_to_base64(real_im)

    @change("real_space_roi")
    def update_diffr(self, *args, **kwargs):
        if not self.loaded:
            return

        self.dp[:] = self.getDenseFrame_jit(
            self.fr_full_3d[
                self.state.real_space_roi[1]:self.state.real_space_roi[1] + self.state.real_space_roi[3] + 1,
                self.state.real_space_roi[0]:self.state.real_space_roi[0] + self.state.real_space_roi[2] + 1,
                :, :
            ],
            self.frame_dimensions)

        diff_c_data = self.apply_colormap(self.dp, self.frame_dimensions, cm.cividis, True)
        diff_im = Image.fromarray(diff_c_data)
        self.state.diff_image = self.convert_to_base64(diff_im)

    # @change('selected_dataset')
    def print_item(self, selected_dataset, **kwargs):
        """ Print out the dataset name when the selection box is changed.
        Uncomment @change to connect this to the selection box and comment out
        elsewhere.
        
        """
        if not selected_dataset:
            print('No dataset selected')
            return

        print('Selected dataset:', selected_dataset)
        if selected_dataset:
            print('File path:', self.state.selected_dataset)

    def _build_ui(self):
        
        with DivLayout(self.server) as layout:
            self._ui = layout

            layout.root.style = "height: 100%;"

            client.Style("""
                            html { height: 100%; overflow: hidden;}
                            body { height: 100%; margin: 0;}
                            #app { height: 100%; }
                         """)
            # Add a selection at the top
            vuetify.VSelect(
                    label='Select Dataset',
                    items=('dataset_names',),
                    v_model=('selected_dataset',)
                    )
            
            with html.Div(style="position: absolute; width: 50%; height: 100%; background-color: black;"):
                with TrameImage(
                    src=("real_image",),
                    size=("real_image_size",),
                    v_model_scale=("real_scale", 0.9),
                    v_model_center=("real_center", [0.5, 0.5]),
                ):
                    TrameImageRoi(v_model=("real_space_roi",),)

                html.Button(
                    "Reset Camera", style="position: absolute; left: 1rem; top: 1rem;",
                    click="real_scale = 0.9; real_center = [0.5, 0.5];"
                )

            with html.Div(style="position: absolute; left: 50%; width: 50%; height: 100%; background-color: black; border-left-style: solid; border-left-color: grey;"):
                with TrameImage(
                    src=("diff_image",),
                    size=("diff_image_size",),
                    v_model_scale=("diff_scale", 0.9),
                    v_model_center=("diff_center", [0.5, 0.5]),
                ):
                    TrameImageRoi(v_model=("diffraction_space_roi",),)

                html.Button(
                    "Reset Camera", style="position: absolute; left: 1rem; top: 1rem;",
                    click="diff_scale = 0.9; diff_center = [0.5, 0.5];"
                )
                    

    def apply_colormap(self, data, shape, colormap, log):
        if log:
            data = np.log(data + 1)

        fdata = np.empty(shape=data.shape, dtype=np.float32)
        min_val = np.min(data)
        max_val = np.max(data)
        delta = max_val - min_val
        fdata[:] = (data[:] - min_val) / delta
        fdata = fdata.reshape(shape)

        return np.uint8(colormap(fdata) * 255)

    @staticmethod
    def convert_to_base64(img: Image.Image) -> str:
        """Convert image to base64 string"""
        buf = BytesIO()
        img.save(buf, format="png")
        return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()

    @change('selected_dataset')
    def setData(self, selected_dataset, **kwargs):
        """ Load the data from the HDF5 file. Must be in
        the format output by stempy.io.save_electron_data().

        Parameters
        ----------
        fPath : pathlib.Path
            The path of to the file to load.
        """
        print(selected_dataset)
        fPath = Path(selected_dataset)
        # Temporary: remove "full expansion" warning
        
        stio.sparse_array._warning = lambda x : None

        # Load data as a SparseArray class
        self.sa = stio.SparseArray.from_hdf5(str(fPath))

        self.sa.allow_full_expand = True
        self.scan_dimensions = self.sa.scan_shape
        self.frame_dimensions = self.sa.frame_shape
        self.num_frames_per_scan = self.sa.num_frames_per_scan
        print('scan dimensions = {}'.format(self.scan_dimensions))

        # Pre-calculate to speed things up
        # Create a non-ragged array with zero padding
        mm = 0
        for ev in self.sa.data.ravel():
            if ev.shape[0] > mm:
                mm = ev.shape[0]
        print('non-ragged array shape: {}'.format((self.sa.data.ravel().shape[0], mm)))

        self.fr_full = np.zeros((self.sa.data.ravel().shape[0], mm), dtype=self.sa.data[0][0].dtype)
        for ii, ev in enumerate(self.sa.data.ravel()):
            self.fr_full[ii, :ev.shape[0]] = ev
        self.fr_full_3d = self.fr_full.reshape((*self.scan_dimensions, self.num_frames_per_scan, self.fr_full.shape[1]))

        print('non-ragged array size = {} GB'.format(self.fr_full.nbytes / 1e9))
        print('Full memory requirement = {} GB'.format(3 * self.fr_full.nbytes / 1e9))

        # Find the row and col for each electron strike
        self.fr_rows = (self.fr_full // int(self.frame_dimensions[0])).reshape(self.scan_dimensions[0] * self.scan_dimensions[1], self.num_frames_per_scan, mm)
        self.fr_cols = (self.fr_full  % int(self.frame_dimensions[1])).reshape(self.scan_dimensions[0] * self.scan_dimensions[1], self.num_frames_per_scan, mm)

        self.dp = np.zeros(self.frame_dimensions[0] * self.frame_dimensions[1], np.uint32)
        self.rs = np.zeros(self.scan_dimensions[0] * self.scan_dimensions[1], np.uint32)

        self.state.real_space_roi[0] = int(self.scan_dimensions[0] // 4 + self.scan_dimensions[0] //8)
        self.state.real_space_roi[1] = int(self.scan_dimensions[1] // 4 + self.scan_dimensions[1] //8)
        self.state.real_space_roi[2] = int(self.scan_dimensions[0] // 4)
        self.state.real_space_roi[3] = int(self.scan_dimensions[1] // 4)
        self.state.real_image_size = list(map(lambda x : int(x), self.scan_dimensions))

        self.state.diffraction_space_roi[0] = int(self.frame_dimensions[0] // 4 + self.frame_dimensions[0] //8)
        self.state.diffraction_space_roi[1] = int(self.frame_dimensions[1] // 4 + self.frame_dimensions[1] //8)
        self.state.diffraction_space_roi[2] = int(self.frame_dimensions[0] // 4)
        self.state.diffraction_space_roi[3] = int(self.frame_dimensions[1] // 4)
        self.state.diff_image_size = list(map(lambda x : int(x), self.frame_dimensions))

        self.loaded = True

        self.update_real()
        self.update_diffr()

    @staticmethod
    @jit(["uint32[:](uint32[:,:,:], uint32[:,:,:], int64, int64, int64, int64)"], nopython=True, nogil=True, parallel=True)
    def getImage_jit(rows, cols, left, right, bot, top):
        """ Sum number of electron strikes within a square box
        significant speed up using numba.jit compilation.

        Parameters
        ----------
        rows : 2D ndarray, (M, num_frames, N)
            The row of the electron strike location. Floor divide by frame_dimenions[0]. M is
            the raveled scan_dimensions axis and N is the zero-padded electron
            strike position location.
        cols : 2D ndarray, (M, num_frames, N)
            The column of the electron strike locations. Modulo divide by frame_dimensions[1]
        left, right, bot, top : int
            The locations of the edges of the boxes

        Returns
        -------
        : ndarray, 1D
            An image composed of the number of electrons for each scan position summed within the boxed region in
        diffraction space.

        """
        
        im = np.zeros(rows.shape[0], dtype=np.uint32)
        
        # For each scan position (ii) sum all events (kk) in each frame (jj)
        for ii in prange(im.shape[0]):
            ss = 0
            for jj in range(rows.shape[1]):
                for kk in range(rows.shape[2]):
                    t1 = rows[ii, jj, kk] > left
                    t2 = rows[ii, jj, kk] < right
                    t3 = cols[ii, jj, kk] > bot
                    t4 = cols[ii, jj, kk] < top
                    t5 = t1 * t2 * t3 * t4
                    if t5:
                        ss += 1
            im[ii] = ss
        return im

    @staticmethod
    @jit(nopython=True, nogil=True, parallel=True)
    #@jit(["uint32[:](uint32[:,:,:,:], UniTuple(int64, 2))"], nopython=True, nogil=True, parallel=True)
    def getDenseFrame_jit(frames, frame_dimensions):
        """ Get a frame summed from the 3D array.

        Parameters
        ----------
        frames : 3D ndarray, (I, J, K, L)
            A set of sparse frames to sum. Each entry is used as the strike location of an electron. I, J, K, L
            corresond to scan_dimension0, scan_dimension1, num_frame, event.
        frame_dimensions : tuple
            The size of the frame

        Returns
        -------
        : ndarray, 2D
        An image composed of the number of electrons in each detector pixel.


        """
        dp = np.zeros((frame_dimensions[0] * frame_dimensions[1]), np.uint32)
        # nested for loop for: scan_dimension0, scan_dimension1, num_frame, event
        for ii in prange(frames.shape[0]):
            for jj in prange(frames.shape[1]):
                for kk in prange(frames.shape[2]):
                    for ll in prange(frames.shape[3]):
                        pos = frames[ii, jj, kk, ll]
                        if pos > 0:
                            dp[pos] += 1
        return dp

def main(server=None, **kwargs):
    app = ExampleApp(server)
    app.server.start(**kwargs)


if __name__ == "__main__":
    main()
