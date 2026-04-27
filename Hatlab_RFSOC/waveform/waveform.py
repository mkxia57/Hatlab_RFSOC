import warnings
from typing import List, Dict, Union, Type, Callable
import numpy as np
from qick.asm_v1 import QickProgram
from qick.qick_asm import QickConfig
import matplotlib.pyplot as plt
from Hatlab_RFSOC.waveform.modulation import ModulationRegistry



NumType = Union[int, float]


class WaveformRegistry:
    _registry = {}

    @classmethod
    def register(cls, shape: str, waveform_cls: Type['Waveform']):
        cls._registry[shape] = waveform_cls

    @classmethod
    def create(cls, shape: str, *args, **kwargs) -> 'Waveform':
        for wave in cls._registry:
            if shape.lower() == wave.lower():
                shape = wave
        if shape not in cls._registry:
            raise ValueError(f"Waveform '{shape}' is not registered.")
        return cls._registry[shape](*args, **kwargs)

    @classmethod
    def available_waveforms(cls):
        return list(cls._registry.keys())


class Waveform:
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        WaveformRegistry.register(cls.__name__, cls)

    def __init__(self, soccfg: QickConfig, gen_ch: Union[int, str], phase, maxv):
        self._set_channel_cfg(soccfg, gen_ch)
        self.maxv = self.soc_gencfg['maxv'] * self.soc_gencfg['maxv_scale'] if maxv is None else maxv
        self.phase = phase
        self.waveform = None

    @staticmethod
    def core(*args, **kwargs) -> np.ndarray:
        pass

    def _generate_waveform(self, *args, **kwargs):
        """
        Generates waveform based on the core function.
        Subclasses should implement this method to generate the waveform.
        """
        raise NotImplementedError("Subclasses must implement _generate_waveform().")

    def add_waveform(self, prog: QickProgram, name):
        idata = self.pad_to_clk_cycle(np.real(self.waveform))
        qdata = self.pad_to_clk_cycle(np.imag(self.waveform))

        if np.max(np.abs(idata)) > 32766 or np.max(np.abs(qdata)) > 32766:
            i_max, q_max = np.max(np.abs(idata)), np.max(np.abs(qdata))
            k = 32766 / np.max((i_max, q_max))
            idata *= k
            qdata *= k
            # warnings.warn("pulse amplitude exceeded maxv")
            print(f"pulse '{name}' amplitude exceeded maxv by {np.max((i_max, q_max)) - 32766}")

        prog.add_pulse(self.gen_ch, name, idata=idata.astype(int), qdata=qdata.astype(int))

    def _set_channel_cfg(self, soccfg: QickConfig, gen_ch: int):
        self.gen_ch = gen_ch
        self.soc_gencfg = soccfg['gens'][gen_ch]
        self.samps_per_clk = self.soc_gencfg['samps_per_clk']
        self.fclk = self.soc_gencfg['f_fabric']
        self.sampling_rate = self.samps_per_clk * self.fclk

    def us_to_samps(self, length):
        """Convert length in physical units to register units."""
        return length * self.sampling_rate

    def pad_to_clk_cycle(self, waveform: np.ndarray) -> np.ndarray:
        """Pad waveform with zeros so that it's length is a multiple of samps_per_clk."""
        pad_len = (-len(waveform)) % self.samps_per_clk
        return np.pad(waveform, (0, pad_len))

    def _apply_padding(self, data: np.ndarray, padding: Union[float, List[float], None]) -> np.ndarray:
        """Pad waveform with zeros before and/or after."""
        if padding is None:
            padding = [0, 0]
        elif isinstance(padding, (int, float)):
            padding = [0, padding]

        padding_reg = np.ceil(self.us_to_samps(np.array(padding))).astype(int)
        return np.pad(data, (padding_reg[0], padding_reg[1]))

    def plot_waveform(self, ax=None, clock_cycle=False):
        """Plots the waveform."""
        fig, ax = plt.subplots() if ax is None else (ax.get_figure(), ax)
        if clock_cycle:
            t_list = np.arange(0, len(self.waveform)) / self.samps_per_clk
            ax.set_xlabel("gen_ch clock cycle")
        else:
            t_list = np.arange(0, len(self.waveform)) / (self.fclk * self.samps_per_clk)
            ax.set_xlabel("Time (us)")
        ax.set_ylabel("Amplitude")
        ax.grid()
        ax.plot(t_list, np.real(self.waveform), label="I")
        ax.plot(t_list, np.imag(self.waveform), label="Q")
        ax.plot(t_list, np.abs(self.waveform), label="mag", linestyle="dashed")
        ax.legend()


class Gaussian(Waveform):
    def __init__(self, soccfg: QickConfig, gen_ch, length, sigma, phase=0, maxv=None,
                 padding: Union[float, List[float], None] = None, modulations: Union[List, tuple] = None,
                 shape=None):
        super().__init__(soccfg, gen_ch, phase=phase, maxv=maxv)
        self.sigma_samps = self.us_to_samps(sigma)
        self.length_samps = self.us_to_samps(length)
        self.padding = padding
        self.modulations = modulations if modulations is not None else []
        self.waveform = self._generate_waveform(self.length_samps, self.sigma_samps)
        # Register custom shape if provided
        if shape is not None:
            WaveformRegistry.register(shape, self.__class__)

    @staticmethod
    def core(length, sigma):
        """the definetion of Gaussian"""
        t = np.arange(length)
        y = np.exp(-(t - length / 2) ** 2 / sigma ** 2)
        return y - np.min(y)

    def _generate_waveform(self, *args, **kwargs):
        """
        apply the necessary modificaiton to the core function,
        generate in-phase (I) and quadrature (Q) components
        """
        waveform = self.maxv * self.core(*args, **kwargs)
        waveform = self._apply_padding(waveform, self.padding)
        waveform = np.exp(1j * np.deg2rad(self.phase)) * waveform
        # Apply modulations if any
        for mod in self.modulations:
            waveform = mod.apply_modulation(waveform, self.sampling_rate)
        return waveform


class TanhBox(Waveform):
    def __init__(self, soccfg: QickConfig, gen_ch, length: float, ramp_width: float, cut_offset=0.01, phase=0, maxv=None,
                 padding: Union[float, List[float], None] = None, modulations: Union[List, tuple] = None,
                 shape=None):
        super().__init__(soccfg, gen_ch, phase=phase, maxv=maxv)
        self.ramp_samps = self.us_to_samps(ramp_width)
        self.length_samps = self.us_to_samps(length)
        self.cut_offset = cut_offset
        self.padding = padding
        self.modulations = modulations if modulations is not None else []
        self.waveform = self._generate_waveform(self.length_samps, self.ramp_samps, self.cut_offset)
        # Register custom shape if provided
        if shape is not None:
            WaveformRegistry.register(shape, self.__class__)

    @staticmethod
    def core(length, ramp_width, cut_offset):
        """
        Create a numpy array containing a smooth box pulse made of two tanh functions subtract from each other.

        :param length: number of points of the pulse
        :param ramp_width: number of points from cutOffset to 0.95 amplitude
        :param cut_offset: the initial offset to cut on the tanh Function
        :return:
        """
        t = np.arange(length)
        c0_, c1_ = np.arctanh(2 * cut_offset - 1), np.arctanh(2 * 0.95 - 1)
        k_ = (c1_ - c0_) / ramp_width
        y = (0.5 * (np.tanh(k_ * t + c0_) - np.tanh(k_ * (t - length) - c0_)) - cut_offset) / (1 - cut_offset)
        return y - np.min(y)

    def _generate_waveform(self, *args, **kwargs):
        """
        apply the necessary modificaiton to the core function,
        generate in-phase (I) and quadrature (Q) components
        """
        waveform = self.maxv * self.core(*args, **kwargs)
        waveform = self._apply_padding(waveform, self.padding)
        waveform = np.exp(1j * np.deg2rad(self.phase)) * waveform
        # Apply modulations if any
        for mod in self.modulations:
            waveform = mod.apply_modulation(waveform, self.sampling_rate)
        return waveform


# Backward compatibility aliases
class GaussianModulated(Gaussian):
    """Alias for Gaussian. Kept for backward compatibility.
    
    Use Gaussian(..., modulations=[...]) instead.
    """
    pass

# Backward compatibility aliases
class TanhBoxModulated(TanhBox):
    """Alias for TanhBox. Kept for backward compatibility.
    
    Use TanhBox(..., modulations=[...]) instead.
    """
    pass


class FileDefined(Waveform):
    def __init__(self, soccfg: QickConfig, gen_ch, filepath, phase=0, maxv=None, drag_coeff=0,
                 padding: Union[float, List[float], None] = None):
        super().__init__(soccfg, gen_ch, phase=phase, maxv=maxv)
        self.filepath = filepath
        self.padding = padding
        self.drag_coeff = drag_coeff
        self.waveform = self._generate_waveform(filepath=self.filepath)

    @staticmethod
    def core(filepath, **kwargs):
        """
        Reads waveform data (I, Q) from the file.
        
        Supports data in two formats:
        - (N, 2): columns are I and Q
        - (2, N): rows are I and Q

        param filepath: the filepath of the waveform
        return: complex waveform
        """
        # todo: deal with file formats
        filetype = filepath.split(".")[-1]
        if filetype == "npy":
            data = np.load(filepath, **kwargs)
        elif filetype == "csv":
            data = np.loadtxt(filepath, delimiter=',', **kwargs)
        else:
            try:
                data = np.loadtxt(filepath, **kwargs)
            except Exception as e:
                raise ValueError(f"Error reading file {filepath}. Exception {e}")
        
        # Handle data shape: (N, 2) means columns are I,Q; (2, N) means rows are I,Q
        data = np.asarray(data)
        if data.ndim == 2:
            if data.shape[1] == 2:
                # Shape (N, 2): columns are I and Q
                idata = data[:, 0]
                qdata = data[:, 1]
            elif data.shape[0] == 2:
                # Shape (2, N): rows are I and Q
                idata = data[0]
                qdata = data[1]
            else:
                raise ValueError(f"Data shape {data.shape} not recognized. Expected (N, 2) or (2, N)")
        else:
            raise ValueError(f"Expected 2D data, got shape {data.shape}")
        
        return idata + 1j * qdata

    def _generate_waveform(self, *args, **kwargs):
        """
        apply the necessary modificaiton to the core function,
        generate in-phase (I) and quadrature (Q) components
        """
        waveform = self.core(*args, **kwargs)
        waveform_padded = self._apply_padding(waveform, self.padding)
        waveform_wphase = np.exp(1j * np.deg2rad(self.phase)) * waveform_padded
        # waveform_dragged = self.apply_drag_modulation(waveform_wphase, drag_coeff=self.drag_coeff)
        return waveform_wphase


class Arbitrary(Waveform):
    # not tested yet.
    """Waveform from arbitrary IQ data.
    
    Accepts IQ data directly as a complex array or separate I and Q arrays.
    """
    def __init__(self, soccfg: QickConfig, gen_ch, iq_data, 
                 phase=0, maxv=None, padding: Union[float, List[float], None] = None,
                 modulations: Union[List, tuple] = None, shape=None):
        """
        Parameters
        ----------
        soccfg : QickConfig
            QickConfig object
        gen_ch : int or str
            Generator channel
        iq_data : array-like
            Complex or IQ data as:
            - Complex array: np.array([1+1j, 2+2j, ...])
            - (I, Q) sequence: (idata, qdata) or [idata, qdata]
            - 2D array shape (N, 2): columns are I and Q
            - Any array-like format convertible via np.asarray()
        phase : float
            Phase in degrees
        maxv : float, optional
            Maximum voltage. If None, uses default from soccfg.
        padding : float or list, optional
            Padding before and/or after waveform in microseconds.
        modulations : list, optional
            List of Modulation objects to apply.
        shape : str, optional
            Custom shape name for registry.
        """
        super().__init__(soccfg, gen_ch, phase=phase, maxv=maxv)
        self.padding = padding
        self.modulations = modulations if modulations is not None else []
        self.waveform = self._generate_waveform(iq_data=iq_data)
        # Register custom shape if provided
        if shape is not None:
            WaveformRegistry.register(shape, self.__class__)

    @staticmethod
    def core(iq_data) -> np.ndarray:
        """Convert IQ data to complex waveform.
        
        Parameters
        ----------
        iq_data : array-like
            Complex or IQ data as:
            - Complex array: np.array([1+1j, 2+2j, ...])
            - (I, Q) sequence: (idata, qdata) or [idata, qdata]
            - 2D array shape (N, 2): columns are I and Q
            - Any other array-like format convertible to complex via np.asarray()
        
        Returns
        -------
        np.ndarray
            Complex waveform
        """
        # Try to handle (I, Q) format for tuple or list
        if isinstance(iq_data, (tuple, list)) and len(iq_data) == 2:
            try:
                idata = np.asarray(iq_data[0])
                qdata = np.asarray(iq_data[1])
                # Verify both are arrays and have compatible shapes
                if idata.shape == qdata.shape and idata.ndim >= 1:
                    return idata + 1j * qdata
            except (TypeError, IndexError, ValueError):
                pass
        
        # Try as generic array-like and check if 2D with shape (N, 2)
        iq_data_arr = np.asarray(iq_data)
        if iq_data_arr.ndim == 2 and iq_data_arr.shape[1] == 2:
            return iq_data_arr[:, 0] + 1j * iq_data_arr[:, 1]
        
        # Otherwise treat as complex array directly
        return iq_data_arr

    def _generate_waveform(self, iq_data: Union[np.ndarray, tuple]):
        """Generate waveform from IQ data with optional modulations."""
        waveform = self.core(iq_data)
        waveform = self._apply_padding(waveform, self.padding)
        waveform = np.exp(1j * np.deg2rad(self.phase)) * waveform
        # Apply modulations if any
        for mod in self.modulations:
            waveform = mod.apply_modulation(waveform, self.sampling_rate)
        return waveform


class ConcatenateWaveform(Waveform):
    def __init__(self, soccfg: QickConfig, gen_ch, waveforms: List[Waveform], phase=0, maxv=None, shape=None):
        super().__init__(soccfg, gen_ch, phase, maxv)
        self.wavefrom_list = waveforms
        self.waveform = self._generate_waveform()
        shape = shape if shape is not None else self.__class__.__name__
        WaveformRegistry.register(shape, self.__class__)

    def _generate_waveform(self):
        return np.concatenate([w.waveform for w in self.wavefrom_list])


def add_waveform(prog: QickProgram, gen_ch, name, shape, **kwargs):
    """Adds a waveform to the DAC channel, using physical parameters of the pulse.
    The pulse will peak at length/2.

    Parameters
    ----------
    prog: QickProgram
        The experiment QickProgram
    gen_ch : str
        name of the generator channel
    name : str
        Name of the pulse
    shape : str
        shape/type of the pulse, e.g. Gaussian, TanhBoxModulated
    """
    if shape.lower() in (wave.lower() for wave in WaveformRegistry.available_waveforms()):
        # pulse = WaveformRegistry.create(shape=shape, prog=prog, gen_ch=gen_ch, **kwargs)
        pulse = WaveformRegistry.create(shape=shape, soccfg=prog.soccfg, gen_ch=gen_ch, **kwargs)
        # pulse.plot_waveform()
        pulse.add_waveform(prog, name=name)
    else:
        raise NameError(f"Unsupported pulse shape {shape}."
                        f"Choose from available shapes: {WaveformRegistry.available_waveforms()},"
                        f"or define new waveforms.")

def add_waveform_from_cfg(prog: QickProgram, gen_ch: str | int, name, **kwargs):
    """
    Add a waveform to the DAC channel based on a configuration dictionary.
    """
    shape = kwargs.get('shape')
    if not shape:
        raise ValueError("cfg_waveform must have a 'shape' key")

    # Create the waveform from the configuration
    waveform = create_waveform_from_config(prog.soccfg, gen_ch, wf_config=kwargs,
                                           modulations_config=prog.cfg.get('modulations', {}))
    waveform.add_waveform(prog, name=name)

def create_waveform_from_config(soccfg: QickConfig, gen_ch: Union[int, str], 
                                wf_config: dict, modulations_config: dict = None) -> Waveform:
    ## Todo: generated by claude, need to be tested and debugged. And think about the best way to integrate with the existing codebase.
    """Create a waveform from config dicts.
    
    Parameters
    ----------
    soccfg : QickConfig
        QickConfig object
    gen_ch : int or str
        Generator channel
    wf_config : dict
        Waveform config dict with 'shape' and waveform-specific parameters.
        Example: {"shape": "TanhBox", "length": 1, "ramp_width": 0.1, "modulations": ["freq_shift"]}
    modulations_config : dict, optional
        Dict mapping modulation names to modulation config dicts.
        Example: {"freq_shift": {"type": "FrequencyConversion", "freq_if": 10}}
    
    Returns
    -------
    Waveform
        Instantiated waveform object
    
    Examples
    --------
    >>> modulations_cfg = {
    ...     "freq_shift": {"type": "FrequencyConversion", "freq_if": 10},
    ...     "drag": {"type": "DragModulation", "drag_factor": 0.05}
    ... }
    >>> wf_cfg = {
    ...     "shape": "TanhBox",
    ...     "length": 1,
    ...     "ramp_width": 0.1,
    ...     "modulations": ["freq_shift", "drag"]
    ... }
    >>> wf = create_waveform_from_config(soccfg, gen_ch=0, wf_config=wf_cfg, modulations_config=modulations_cfg)
    """
    
    modulations_config = modulations_config or {}
    wf_config = wf_config.copy()
    
    # Extract shape and modulation names
    shape = wf_config.pop('shape', None)
    if shape is None:
        raise ValueError("wf_config must have a 'shape' key")
    
    modulation_names = wf_config.pop('modulations', [])
    
    # Create modulation objects from config
    modulations = []
    for mod_name in modulation_names:
        if mod_name not in modulations_config:
            raise ValueError(f"Modulation '{mod_name}' not found in config['modulations']. "
                           f"Available: {list(modulations_config.keys())}")
        mod_config = modulations_config[mod_name]
        mod = ModulationRegistry.from_dict(mod_config)
        modulations.append(mod)
    
    # Create waveform with modulations
    wf_config['modulations'] = modulations
    waveform = WaveformRegistry.create(shape=shape, soccfg=soccfg, gen_ch=gen_ch, **wf_config)
    return waveform

def add_waveform_concatenate(prog: QickProgram, gen_ch: str | int, name, gatelist: List[Dict], maxv=None):
    """Concatenate a list of waveforms defined by gatelist and add to the DAC channel.
    
    gatelist is a list of dict, each dict contains the parameters to define a waveform, 
    including 'shape' and waveform-specific parameters.
    
    Parameters
    ----------
    prog : QickProgram
        The experiment QickProgram
    gen_ch : int or str
        Generator channel
    name : str
        Name of the concatenated pulse
    gatelist : list of dict
        List of waveform config dicts. Each dict must have 'shape' and waveform parameters.
        Example: [
            {"shape": "TanhBox", "length": 0.05, "ramp_width": 0.01},
            {"shape": "Gaussian", "length": 0.05, "sigma": 0.01}
        ]
    maxv : float, optional
        Maximum voltage for the concatenated waveform.
    
    Examples
    --------
    >>> gatelist = [
    ...     {"shape": "TanhBox", "length": 0.05, "ramp_width": 0.01, "padding": [0.01, 0.01]},
    ...     {"shape": "Gaussian", "length": 0.05, "sigma": 0.01, "padding": [0.01, 0.01]}
    ... ]
    >>> add_waveform_concatenate(prog, gen_ch=0, name="concat_pulse", gatelist=gatelist)
    """
    if not gatelist:
        raise ValueError("gatelist cannot be empty")
    
    # Get modulations config if available
    modulations_config = prog.cfg.get('modulations', {}) if hasattr(prog, 'cfg') else {}
    
    # Create waveforms from each config
    waveforms = []
    for wf_config in gatelist:
        wf = create_waveform_from_config(prog.soccfg, gen_ch, wf_config=wf_config, 
                                         modulations_config=modulations_config)
        waveforms.append(wf)
    
    # Create concatenated waveform
    concat_wf = ConcatenateWaveform(prog.soccfg, gen_ch, waveforms=waveforms, maxv=maxv)
    
    # Add to program
    concat_wf.add_waveform(prog, name=name)


if __name__ == "__main__":
    from Hatlab_RFSOC.core.averager_program import NDAveragerProgram, QubitMsmtMixin
    from Hatlab_RFSOC.proxy import getSocProxy
    from Hatlab_RFSOC.waveform import modulation
    import yaml

    # --------------------- get qick config -------------------------------------------------------
    soc, soccfg = getSocProxy("pynq216-04")

    # --------------------- generate waveforms ----------------------------------------------------
    # wf = Gaussian(prog, 0, length=0.05, sigma=0.01, phase=0, padding=[0.05, 0.05])
    wf = TanhBox(soccfg, 0, length=0.1, ramp_width=0.01, phase=0, padding=[0.015, 0.015])
    wf.plot_waveform()
    plt.ylim((-35000, 35000))
    plt.tight_layout()

    # --------------------- generate modulated waveforms ----------------------------------------------------
    # define modulations
    def cfunc(amp, maxf, maxv):
        return maxf * (amp/maxv)**2
    cm = modulation.ChirpModulation(chirp_func=cfunc, maxf=-50, maxv=30000)
    # dm = modulation.DragModulation(0.00)
    dm = modulation.ModulationRegistry.create("DragModulation", drag_factor=0.001)

    # wf2 = GaussianModulated(soccfg=soccfg, gen_ch=0, length=0.05, sigma=0.01, phase=0, padding=[0.015, 0.015],
    #                         modulations=[dm, cm], shape="GaussianChirpDrag")
    wf2 = TanhBoxModulated(soccfg=soccfg, gen_ch=0, length=0.05, ramp_width=0.01, phase=0, padding=[0.015, 0.015],
                           modulations=[dm, cm], shape="TanhboxChirpDrag")
    wf2.plot_waveform()

    wfc = ConcatenateWaveform(soccfg=soccfg, gen_ch=0, waveforms=[wf, wf2], phase=0, shape="concatenated_pulse_1")
    wfc.plot_waveform()

    corrFile = r"W:\data\SubHarmonic\WileE_20250326\Q1\calibration\\" \
               r"Q1_DAC0_Line1_Subh1000-1300MHz_5000DAC_Q3700-3720MHz_8000DAC-2_epsilon_smoothed(2).csv"
    wcorr = modulation.WaveformCorrection(filepath=corrFile, freq=1100, scale="linear", max_scale=0.5)
    wfcorr = TanhBoxModulated(soccfg=soccfg, gen_ch=0, length=0.1, ramp_width=0.01, phase=0, padding=[0.05, 0.05],
                           modulations=[cm, wcorr], shape="TanhboxCorrected")
    wfcorr = TanhBoxModulated(soccfg=soccfg, gen_ch=0, length=0.1, ramp_width=0.01, phase=0, padding=[0.015, 0.015],
                           modulations=[wcorr], shape="TanhboxCorrected")
    # wfcorr = GaussianModulated(soccfg=soccfg, gen_ch=0, length=0.1, sigma=0.02, phase=0, padding=[0.015, 0.015],
    #                        modulations=[wcorr], shape="GaussianCorrected")
    wfcorr.plot_waveform()
    plt.ylim((-35000, 35000))
    plt.tight_layout()

    wf_fft = wcorr.compute_fourier_transform(wfcorr.waveform, wfcorr.sampling_rate)
    plt.figure()
    plt.plot(wf_fft[0], wf_fft[1])

    t_list = np.linspace(0, (len(wf.waveform) - 1) / wf.sampling_rate, len(wf.waveform))
    signal = wf.waveform
    signal_wc = wcorr.apply_modulation(signal, wf.sampling_rate)
    signal_recv = wcorr.recover_modulation(signal_wc, wf.sampling_rate)

    plt.figure()
    plt.plot(t_list, np.abs(signal), label="original")
    plt.plot(t_list, np.abs(signal_wc), label="corrected")
    plt.plot(t_list, np.abs(signal_recv), label="undo")
    plt.legend()

    fconv = modulation.FrequencyConversion(-1000)
    wfcorr = TanhBoxModulated(soccfg=soccfg, gen_ch=0, length=0.1, ramp_width=0.01, phase=0, padding=[0.025, 0.025],
                           modulations=[fconv], shape="TanhboxShifted")
    wf_fft = wcorr.compute_fourier_transform(wfcorr.waveform, wfcorr.sampling_rate)
    plt.figure()
    plt.plot(wf_fft[0], wf_fft[1])
    plt.figure()
    plt.plot(np.real(wfcorr.waveform))


