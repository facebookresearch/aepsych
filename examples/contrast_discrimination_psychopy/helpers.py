import numpy as np
import pyglet
from psychopy import core, event
from psychopy.visual import Window

from psychopy.visual.image import ImageStim

pyglet.options["debug_gl"] = False
GL = pyglet.gl


def polar_to_cartesian(r, theta):
    z = r * np.exp(1j * np.radians(theta))
    return z.real, z.imag


def cartesian_to_polar(x, y):
    z = x + 1j * y
    return (np.abs(z), np.angle(z, deg=True))


class AnimatedGrating:
    param_transforms = {"contrast": lambda x: 10**x, "pedestal": lambda x: 10**x}

    def __init__(
        self,
        spatial_frequency: float,
        orientation: float,
        pedestal: float,
        contrast: float,
        temporal_frequency: float,
        eccentricity: float,
        size: float,
        angle_dist: float,
        win: Window,
        cpd=60,  # display cycles per degree
        Lmin=0,  # min luminance in nits
        Lmax=255,  # max luminance in nits
        res=256,  # texture resolution
        noisy=False,
        *args,
        **kw,
    ):
        """Generate animated Gabor grating

        Args:
            spatial_frequency (float): Spatial frequency.
            orientation (float): Orientation (degrees)
            pedestal (float): Background luminance.
            contrast (float): Stimulus contrast.
            temporal_frequency (float): Temporal frequency (seconds).
            eccentricity (float): Stimulus eccentricity relative to center (degrees).
            size (float): Stimulus size.
            angle_dist (float): Stimulus angle relative to center.
            win (Window): Window to render to.
            cpd (int, optional): Display cycles per degree. Defaults to 60.
        """
        self.spatial_frequency = spatial_frequency
        self.temporal_frequency = temporal_frequency
        self.orientation = orientation
        self.pedestal = pedestal
        self.contrast = contrast
        self.settable_params = (
            "spatial_frequency",
            "temporal_frequency",
            "orientation",
            "pedestal",
            "contrast",
            "size",
            "eccentricity",
            "angle_dist",
        )
        self.cpd = cpd
        self.Lmin = Lmin
        self.Lmax = Lmax
        self.res = res
        self.noisy = noisy
        self.initial_phase = np.random.uniform(low=0, high=0.2, size=(1))
        img = np.zeros((self.res, self.res))
        self.win = win
        self._stim = ImageStim(image=img, mask="gauss", win=win, *args, **kw)
        # these get set on _stim
        self.size = size
        self.eccentricity = eccentricity
        self.angle_dist = angle_dist

    def update(self, trial_config):
        for k, v in trial_config.items():
            if k in self.settable_params:
                if k in self.param_transforms:
                    setattr(self, k, self.param_transforms[k](v[0]))
                else:
                    setattr(self, k, v[0])

    @property
    def size(self):
        return self._stim.size

    @size.setter
    def size(self, x):
        self._stim.size = x

    @property
    def eccentricity(self):
        return cartesian_to_polar(*self._stim.pos)[0]

    @eccentricity.setter
    def eccentricity(self, x):
        current_coords = cartesian_to_polar(*self._stim.pos)
        self._stim.pos = polar_to_cartesian(x, current_coords[1])

    @property
    def angle_dist(self):
        return cartesian_to_polar(*self._stim.pos)[1]

    @angle_dist.setter
    def angle_dist(self, deg):
        current_coords = cartesian_to_polar(*self._stim.pos)
        self._stim.pos = polar_to_cartesian(current_coords[0], deg + 90)

    @property
    def pedestal_psychopy_scale(self):
        return self.pedestal * 2 - 1

    def draw(
        self,
        noisy=False,
        win=None,
        pre_duration_s=0.1,
        stim_duration_s=5.0,
        *args,
        **kwargs,
    ):
        win = win or self.win
        clock = core.Clock()
        clock.reset()
        self._stim.image = self.get_texture(self.initial_phase, noisy=noisy)
        while clock.getTime() < pre_duration_s:
            win.flip()

        start_time = clock.getTime()

        while clock.getTime() < pre_duration_s + stim_duration_s:
            if self.temporal_frequency > 0:
                newphase = (clock.getTime() - start_time) * self.temporal_frequency
                self._stim.image = self.get_texture(
                    newphase + self.initial_phase, noisy=noisy
                )
            self._stim.draw()

    def get_texture(self, phase=0, noisy=False):
        pedestal_lum = self.pedestal * (self.Lmax - self.Lmin) + self.Lmin
        grating_max = (self.contrast * (2 * pedestal_lum + self.Lmin) + self.Lmin) / 2
        x = np.arange(0, self.res) / self.cpd + phase
        y = np.arange(0, self.res) / self.cpd + phase
        x_grid, y_grid = np.meshgrid(x, y)
        wave = x_grid * np.cos(np.radians(self.orientation)) + y_grid * np.sin(
            np.radians(self.orientation)
        )
        scaled_imag_wave = 1j * 2 * np.pi * self.spatial_frequency * wave

        img = grating_max * np.real(np.exp(scaled_imag_wave)) + pedestal_lum
        # convert from luminance to values in [-1, 1] as psychopy wants
        img = img / ((self.Lmax - self.Lmin) / 2) - 1

        if noisy:
            flatimg = img.flatten()
            np.random.shuffle(flatimg)
            img = flatimg.reshape(self.res, self.res)
        return img


class HalfGrating(AnimatedGrating):
    """Gabor animated grating, half of which is scrambled into white noise."""

    def noisify_half_texture(self, img, noisy_half):
        img = img.T  # transpose so our indexing tricks work
        flatimg = img.flatten()
        if noisy_half == "left":
            noisy = flatimg[: (self.res**2) // 2]
            np.random.shuffle(noisy)
            img = np.r_[noisy, flatimg[(self.res**2) // 2 :]].reshape(
                self.res, self.res
            )
        else:
            noisy = flatimg[(self.res**2) // 2 :]
            np.random.shuffle(noisy)
            img = np.r_[flatimg[: (self.res**2) // 2], noisy].reshape(
                self.res, self.res
            )
        return img.T  # untranspose

    def get_texture(self, phase, noisy_half):
        img = super().get_texture(phase, noisy=False)
        img = self.noisify_half_texture(img, noisy_half)
        return img

    def draw(
        self,
        noisy_half="left",
        win=None,
        pre_duration_s=0.1,
        stim_duration_s=5.0,
        *args,
        **kwargs,
    ):
        win = win or self.win
        clock = core.Clock()
        clock.reset()
        event.clearEvents()
        self._stim.image = self.get_texture(self.initial_phase, noisy_half=noisy_half)
        while clock.getTime() < pre_duration_s:
            win.flip()

        start_time = clock.getTime()

        while True:
            if self.temporal_frequency > 0:
                newphase = (clock.getTime() - start_time) * self.temporal_frequency
                self._stim.image = self.get_texture(
                    newphase + self.initial_phase, noisy_half=noisy_half
                )
            self._stim.draw()
            keys = event.getKeys(keyList=["left", "right"])
            win.flip()
            if len(keys) > 0:
                return keys
        return keys


class ExperimentAborted(Exception):
    pass


class QuitHelper:
    """Helper to quit the experiment by pressing a key twice within 500ms.
    It quits by simply raising 'ExperimentAborted'. This is necessary because
    from the separate thread that psychopy checks its global key events in, you
    cannot raise an Exception in the main thread.
    """

    def __init__(self):
        self.quit_requested = False
        self.debounce_timestamp = None

    def request_quit(self):
        """Must be called twice in 500ms to set a flag that causes ExperimentAborted
        to be raised when quit_if_requested is called. This indirection is needed if request_quit
        is called from a separate thread (as with psychopy global event keys)
        """
        tprev = self.debounce_timestamp
        tnow = core.getTime()
        if tprev is not None and tnow - tprev < 0.5:
            self.quit_requested = True
        self.debounce_timestamp = tnow

    def quit_if_requested(self):
        """Raises ExperimentAborted if request_quit has been called twice in 500ms"""
        if self.quit_requested:
            raise ExperimentAborted
        return True
