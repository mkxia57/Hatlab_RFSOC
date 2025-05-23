import warnings
from typing import List, Union, Type, Callable
import numpy as np
from qick.asm_v1 import QickProgram

NumType = Union[int, float]


def tanh_box(length: int, ramp_width: int, cut_offset=0.01, maxv=30000):
    """
    Create a numpy array containing a smooth box pulse made of two tanh functions subtract from each other.

    :param length: Length of array (in points)
    :param ramp_width: number of points from cutOffset to 0.95 amplitude
    :param cut_offset: the initial offset to cut on the tanh Function
    :param maxv: the max value of the waveform
    :return:
    """
    x = np.arange(0, length)
    c0_ = np.arctanh(2 * cut_offset - 1)
    c1_ = np.arctanh(2 * 0.95 - 1)
    k_ = (c1_ - c0_) / ramp_width
    y = (0.5 * (np.tanh(k_ * x + c0_) - np.tanh(k_ * (x - length) - c0_)) - cut_offset) / (
            1 - cut_offset) * maxv
    return y - np.min(y)


def gaussian(sigma: int, length: int, maxv=30000):
    """
    Create a numpy array containing a Gaussian function.

    :param sigma: sigma (standard deviation) of Gaussian
    :param length: total number of points of gaussian pulse
    :param maxv: the max value of the waveform
    :return:
    """
    x = np.arange(0, length)
    y = maxv * np.exp(-(x - length / 2) ** 2 / sigma ** 2)
    y = y - np.min(y)
    return y


def tanh_box_fm(freq: float, length: int, ramp_width: int, cut_offset=0.01, maxv=30000):
    x = np.arange(0, length)
    y = tanh_box(length, ramp_width, cut_offset, maxv) * np.cos(2*np.pi * freq * x)
    return y


def tanh_box_IQ(freq: float, length: int, ramp_width: int, cut_offset=0.01, maxv=30000):
    x = np.arange(0, length)
    i = tanh_box(length, ramp_width, cut_offset, maxv) * np.cos(2*np.pi * freq * x)
    q = tanh_box(length, ramp_width, cut_offset, maxv) * np.sin(2*np.pi * freq * x)
    return [i, q]


def gaussian_fm(freq: float, sigma: int, length: int, maxv=30000):
    x = np.arange(0, length)
    y = gaussian(sigma, length, maxv) * np.cos(2*np.pi * freq * x)
    return y


def add_padding(data, soc_gencfg, padding):
    """
    pad some zeros before and/or after the waveform data
    :param data:
    :param soc_gencfg: gen_ch config
    :param padding: the length of padding in us
    :return:
    """
    samps_per_clk = soc_gencfg['samps_per_clk']
    fclk = soc_gencfg['f_fabric']

    if isinstance(padding, int | float):
        padding = np.array([0, padding])
    else:
        padding = np.array(padding)
    padding_samp = np.ceil(padding * fclk * samps_per_clk)
    data = np.concatenate((np.zeros(int(padding_samp[0])), data, np.zeros(int(padding_samp[1]))))

    return data


def add_tanh(prog: QickProgram, gen_ch, name, length: float, ramp_width: float, cut_offset: float = 0.01,
             phase: float = 0, maxv=None, padding: Union[NumType, List[NumType]] = None, drag: float = 0):
    """
    Adds a smooth box pulse made of two tanh functions to the waveform library, using physical parameters of the pulse.
    The pulse will peak at length/2.

    Parameters
    ----------
    gen_ch : str
        name of the DAC channel defined in the YAML
    name : str
        Name of the pulse
    length : float
        Total pulse length (in units of us)
    ramp_width : float
        ramping time from cut_offset to 0.95 amplitude (in units of us)
    cut_offset: float
        the initial offset to cut on the tanh Function (in unit of unit-height pulse)
    phase: float
        the phase of the waveform in degree
    maxv : float
        Value at the peak (if None, the max value for this generator will be used)
    padding: float | List[float]
        padding zeros in front of and at the end of the pulse

    """

    gen_ch = prog.cfg["gen_chs"][gen_ch]["ch"]
    soc_gencfg = prog.soccfg['gens'][gen_ch]
    if maxv is None:
        maxv = soc_gencfg['maxv'] * soc_gencfg['maxv_scale']
    samps_per_clk = soc_gencfg['samps_per_clk']
    fclk = soc_gencfg['f_fabric']

    # length_cyc = prog.us2cycles(length, gen_ch=gen_ch)
    length_cyc = length * fclk
    length_reg = length_cyc * samps_per_clk
    # ramp_reg = np.int64(np.round(ramp_width*fclk*samps_per_clk))
    ramp_reg = ramp_width * fclk * samps_per_clk

    wf = tanh_box(length_reg, ramp_reg, cut_offset, maxv=maxv)
    if padding is not None:
        wf = add_padding(wf, soc_gencfg, padding)
    zero_padding = np.zeros((16 - len(wf)) % 16)
    wf_padded = np.concatenate((wf, zero_padding))
    drag_padded = -np.gradient(wf_padded) * drag

    wf_idata = np.cos(np.pi / 180 * phase) * wf_padded - np.sin(np.pi / 180 * phase) * drag_padded
    wf_qdata = np.sin(np.pi / 180 * phase) * wf_padded + np.cos(np.pi / 180 * phase) * drag_padded

    # prog.add_pulse(gen_ch, name, idata=wf_padded)
    prog.add_pulse(gen_ch, name, idata=wf_idata, qdata=wf_qdata)


def add_gaussian(prog: QickProgram, gen_ch: str, name, sigma: float, length: float, phase: float = 0,
                 maxv=None, padding: Union[NumType, List[NumType]] = None, drag: float = 0):
    """Adds a gaussian pulse to the waveform library, using physical parameters of the pulse.
    The pulse will peak at length/2.

    Parameters
    ----------
    gen_ch : str
        name of the generator channel
    name : str
        Name of the pulse
    sigma : float
        sigma of gaussian (in units of us)
    length : float
        Total pulse length (in units of us)
    maxv : float
        Value at the peak (if None, the max value for this generator will be used)

    """

    gen_ch = prog.cfg["gen_chs"][gen_ch]["ch"]
    soc_gencfg = prog.soccfg['gens'][gen_ch]
    if maxv is None:
        maxv = soc_gencfg['maxv'] * soc_gencfg['maxv_scale']
    samps_per_clk = soc_gencfg['samps_per_clk']
    fclk = soc_gencfg['f_fabric']

    # length_cyc = prog.us2cycles(length, gen_ch=gen_ch)
    length_cyc = length * fclk
    length_reg = length_cyc * samps_per_clk
    sigma_reg = sigma * fclk * samps_per_clk

    wf = gaussian(sigma_reg, length_reg, maxv=maxv)
    if padding is not None:
        wf = add_padding(wf, soc_gencfg, padding)
    zero_padding = np.zeros((16 - len(wf)) % 16)
    wf_padded = np.concatenate((wf, zero_padding))
    drag_padded = -np.gradient(wf_padded) * drag

    wf_idata = np.cos(np.pi / 180 * phase) * wf_padded - np.sin(np.pi / 180 * phase) * drag_padded
    wf_qdata = np.sin(np.pi / 180 * phase) * wf_padded + np.cos(np.pi / 180 * phase) * drag_padded

    # prog.add_pulse(gen_ch, name, idata=wf_padded)
    prog.add_pulse(gen_ch, name, idata=wf_idata, qdata=wf_qdata)


def add_arbitrary(prog: QickProgram, gen_ch: str, name, envelope, phase: float = 0,
                  maxv=None, padding: Union[NumType, List[NumType]] = None, drag: float = 0):
    """Adds an arbitrary pulse to the waveform library, using physical parameters of the pulse.
    The pulse will peak at length/2.

    Parameters
    ----------
    gen_ch : str
        name of the generator channel
    name : str
        Name of the pulse
    envelope : float
        the envelope of the waveform
    maxv : float
        Value at the peak (if None, the max value for this generator will be used)

    """    
    gen_ch = prog.cfg["gen_chs"][gen_ch]["ch"]
    soc_gencfg = prog.soccfg['gens'][gen_ch]

    wf = envelope

    if padding is not None:
        wf = add_padding(wf, soc_gencfg, padding)
    zero_padding = np.zeros((16 - len(wf)) % 16)
    wf_padded = np.concatenate((wf, zero_padding))
    drag_padded = -np.gradient(wf_padded) * drag

    wf_idata = np.cos(np.pi / 180 * phase) * drag_padded
    wf_qdata = np.sin(np.pi / 180 * phase) * drag_padded

    # prog.add_pulse(gen_ch, name, idata=wf_padded)
    prog.add_pulse(gen_ch, name, idata=wf_idata, qdata=wf_qdata)


def add_pulse_concatenate(prog: QickProgram, gen_ch: str | int, name, gatelist, maxv=None):
    def get_gain_max(gatelist):
        gmax = 0
        for gate in gatelist:
            gmax = gate['gain'] if gate['gain'] > gmax else gmax
        return gmax
    gen_ch = prog.cfg["gen_chs"][gen_ch]["ch"] if type(gen_ch) == str else gen_ch
    soc_gencfg = prog.soccfg['gens'][gen_ch]
    if maxv is None:
        maxv = soc_gencfg['maxv'] * soc_gencfg['maxv_scale']
    samps_per_clk = soc_gencfg['samps_per_clk']
    fclk = soc_gencfg['f_fabric']

    wfdata_i = []
    wfdata_q = []
    wf_len_list = []
    gmax = get_gain_max(gatelist)
    for gate in gatelist:
        maxv_p = gate.get('maxv', maxv)
        if gate['shape'] == 'gaussian':
            length_reg = gate['length'] * fclk * samps_per_clk
            sigma_reg = gate['sigma'] * fclk * samps_per_clk
            pulsedata = gate['gain'] / gmax * gaussian(sigma_reg, length_reg, maxv=maxv_p)

        elif gate['shape'] == 'tanh_box':
            length_reg = gate['length'] * fclk * samps_per_clk
            ramp_reg = gate['ramp_width'] * fclk * samps_per_clk
            pulsedata = gate['gain'] / gmax * tanh_box(length_reg, ramp_reg, maxv=maxv_p)

        else:
            raise NameError(f"unsupported pulse shape {gate['shape']}")

        padding = gate.get('padding')
        if gate['padding'] is not None:
            pulsedata = add_padding(pulsedata, soc_gencfg, padding)

        wfdata_i = np.concatenate((wfdata_i, pulsedata * np.cos(gate['phase'] / 360 * 2 * np.pi)))
        wfdata_q = np.concatenate((wfdata_q, pulsedata * np.sin(gate['phase'] / 360 * 2 * np.pi)))
        zero_padding = np.zeros((16 - len(wfdata_i)) % 16)
        wfdata_i = np.concatenate((wfdata_i, zero_padding))
        wfdata_q = np.concatenate((wfdata_q, zero_padding))
        # print("gate phase: ", gate['phase'])
        # wf_len_list.append(len(wfdata_i) / 16.0)

    # print(wf_len_list)
    if len(wfdata_i) == 0:
        prog.add_pulse(gen_ch, name, idata=3 * [0] * samps_per_clk, qdata=3 * [0] * samps_per_clk)
    else:
        prog.add_pulse(gen_ch, name, idata=wfdata_i, qdata=wfdata_q)


# class WaveformRegistry:
#     _registry = {}
# 
#     @classmethod
#     def register(cls, shape: str, waveform_cls: Type['Waveform']):
#         cls._registry[shape] = waveform_cls
# 
#     @classmethod
#     def create(cls, shape: str, *args, **kwargs) -> 'Waveform':
#         for wave in cls._registry:
#             if shape.lower() == wave.lower():
#                 shape = wave
#         if shape not in cls._registry:
#             raise ValueError(f"Waveform '{shape}' is not registered.")
#         return cls._registry[shape](*args, **kwargs)
# 
#     @classmethod
#     def available_waveforms(cls):
#         return list(cls._registry.keys())
# 
# 
# class Waveform:
#     def __init_subclass__(cls, **kwargs):
#         super().__init_subclass__(**kwargs)
#         WaveformRegistry.register(cls.__name__, cls)
# 
#     def __init__(self, prog: QickProgram, gen_ch: Union[int, str], phase, maxv):
#         self._set_channel_cfg(prog, gen_ch)
#         self.maxv = self.soc_gencfg['maxv'] * self.soc_gencfg['maxv_scale'] if maxv is None else maxv
#         self.phase = phase
#         self.waveform = None
# 
#     @staticmethod
#     def core(*args, **kwargs) -> np.ndarray:
#         pass
# 
#     def _generate_waveform(self, *args, **kwargs):
#         """
#         Generates waveform based on the core function.
#         Subclasses should implement this method to generate the waveform.
#         """
#         raise NotImplementedError("Subclasses must implement _generate_waveform().")
# 
#     def add_waveform(self, prog: QickProgram, name):
#         idata = self._pad_waveform(np.real(self.waveform))
#         qdata = self._pad_waveform(np.imag(self.waveform))
# 
#         if np.max(np.abs(idata)) > 32766 or np.max(np.abs(qdata)) > 32766:
#             i_max, q_max = np.max(np.abs(idata)), np.max(np.abs(qdata))
#             k = 32766 / np.max((i_max, q_max))
#             idata *= k
#             qdata *= k
#             # warnings.warn("pulse amplitude exceeded maxv")
#             print(f"pulse '{name}' amplitude exceeded maxv by {np.max((i_max, q_max)) - 32766}")
# 
#         prog.add_pulse(self.gen_ch, name, idata=idata.astype(int), qdata=qdata.astype(int))
# 
#     def _set_channel_cfg(self, prog: QickProgram, gen_ch: Union[int, str]):
#         self.gen_ch = prog.cfg["gen_chs"][gen_ch]["ch"] if isinstance(gen_ch, str) else gen_ch
#         self.soc_gencfg = prog.soccfg['gens'][self.gen_ch]
#         self.samps_per_clk = self.soc_gencfg['samps_per_clk']
#         self.fclk = self.soc_gencfg['f_fabric']
#         self.sampling_rate = self.samps_per_clk * self.fclk
# 
#     def us_to_samps(self, length):
#         """Convert length in physical units to register units."""
#         return length * self.sampling_rate
# 
#     def _pad_waveform(self, waveform: np.ndarray) -> np.ndarray:
#         """Pad waveform with zeros so that it's length is a multiple of samps_per_clk."""
#         pad_len = (-len(waveform)) % self.samps_per_clk
#         return np.pad(waveform, (0, pad_len))
# 
#     def _apply_padding(self, data: np.ndarray, padding: Union[float, List[float], None]) -> np.ndarray:
#         """Pad waveform with zeros before and/or after."""
#         if padding is None:
#             padding = [0, 0]
#         elif isinstance(padding, (int, float)):
#             padding = [0, padding]
# 
#         padding_reg = np.ceil(self.us_to_samps(np.array(padding))).astype(int)
#         return np.pad(data, (padding_reg[0], padding_reg[1]))
# 
#     def plot_waveform(self, ax=None, clock_cycle=False):
#         """Plots the waveform."""
#         fig, ax = plt.subplots() if ax is None else (ax.get_figure(), ax)
#         if clock_cycle:
#             t_list = np.arange(0, len(self.waveform)) / self.samps_per_clk
#             ax.set_xlabel("gen_ch clock cycle")
#         else:
#             t_list = np.arange(0, len(self.waveform)) / (self.fclk * self.samps_per_clk)
#             ax.set_xlabel("Time (us)")
#         ax.set_ylabel("Amplitude")
#         ax.grid()
#         ax.plot(t_list, np.real(self.waveform), label="I")
#         ax.plot(t_list, np.imag(self.waveform), label="Q")
#         ax.plot(t_list, np.abs(self.waveform), label="mag", linestyle="dashed")
#         ax.legend()
# 
# 
# class DragModulationMixin:
#     @staticmethod
#     def _drag_func(waveform, drag_coeff: float = 0, sampling_rate=1):
#         """drag correction for resonant driving"""
#         dt = 1/sampling_rate
#         return -drag_coeff * np.exp(1j * np.pi/2) * np.gradient(waveform, dt)
# 
#     @staticmethod
#     def apply_drag_modulation(waveform, drag_func=None, drag_coeff: float = 0, sampling_rate=1):
#         """
#         Apply a drag modulation to the input waveform.
# 
#         Parameters:
#         - waveform: Input waveform array.
#         - drag_func: A callable accepting (waveform, drag_coeff, sampling_rate) that returns the drag correction.
#         - drag_coeff: A coefficient for the drag correction amplitude
#         """
#         if drag_func is None:
#             drag_func = DragModulationMixin._drag_func
#         wf_drag = drag_func(waveform, drag_coeff, sampling_rate)
#         return waveform + wf_drag
# 
# 
class ChirpModulationMixin:
    @staticmethod
    def _instant_frequency(chirp_func, waveform, maxf, maxv):
        return chirp_func(np.abs(waveform), maxf, maxv)

    @staticmethod
    def _chirp_phase(instant_freq, sampling_rate):
        phase = np.zeros(len(instant_freq))
        phi0 = 0
        for i in range(len(instant_freq) - 1):
            phase[i] = phi0
            phi0 += np.pi * (instant_freq[i] + instant_freq[i + 1]) / sampling_rate
        phase[-1] = phi0
        return phase

    @staticmethod
    def apply_chirp_modulation(waveform, chirp_func, sampling_rate, maxf=0, maxv=30000):
        """
        Apply a chirp modulation to the input waveform.

        Parameters:
        - waveform: Input waveform array.
        - chirp_func: A callable accepting (amp, maxf, maxv) that returns the chirp frequency at the given amplitude.
        - sampling_rate: sampling_rate of the waveform
        - maxf: Maximum chirp instanteous frequency in MHz. Default value is 0
        - maxv: Maximum amplitude of the given waveform. Default value is 30000
        """
        chirp_freq = ChirpModulationMixin._instant_frequency(chirp_func, waveform, maxf, maxv)
        chirp_phase = ChirpModulationMixin._chirp_phase(chirp_freq, sampling_rate)
        wf_chirp = waveform * np.exp(1j * chirp_phase)

        return wf_chirp
# 
# 
# class WaveformCorrectionMixin:
#     @staticmethod
#     def compute_fourier_transform(signal, sampling_rate):
#         """
#         Compute the Fourier Transform of a signal.
# 
#         Parameters:
#         - signal: numpy array of the waveform data.
#         - sampling_rate: sampling rate in Hz.
# 
#         Returns:
#         - frequencies: numpy array of frequency bins (MHz).
#         - magnitudes: numpy array of corresponding magnitude spectrum.
#         """
#         N = len(signal)
#         fft_vals = np.fft.fft(signal)
#         fft_freq = np.fft.fftfreq(N, d=1 / sampling_rate)
# 
#         return fft_freq, fft_vals
# 
#     @staticmethod
#     def compute_inverse_fourier_transform(fft_signal):
#         """
#         Compute the Inverse Fourier Transform of a complex frequency-domain signal.
# 
#         Parameters:
#           fft_signal: numpy array of complex Fourier coefficients (full spectrum).
# 
#         Returns:
#           time_signal: numpy array representing the recovered complex time-domain signal.
#         """
#         time_signal = np.fft.ifft(fft_signal)
#         return time_signal
# 
#     @staticmethod
#     def calibrate_waveform_in_frequency_domain(signal, calibration_func, sampling_rate):
#         """
#         Modify a waveform in the frequency domain and return the modified time-domain waveform.
# 
#         Parameters:
#             - signal (np.array): Input time-domain signal (can be complex or real).
#             - calibration_func (callable): A function that takes (fft_freq, fft_values) as inputs
#                                           and returns modified fft_values.
#             - sampling_rate (float): Sampling rate of the signal in Hz.
# 
#         Returns:
#             np.array: The modified time-domain signal (complex if the input was complex).
#         """
#         N = len(signal)
#         # Compute the Fourier Transform and associated frequency bins.
#         fft_values = np.fft.fft(signal)
#         fft_freq = np.fft.fftfreq(N, d=1 / sampling_rate)
# 
#         # Apply the provided modification function to the FFT coefficients.
#         modified_fft_values = calibration_func(fft_freq) * fft_values
# 
#         # Compute the inverse FFT to get back the modified time-domain waveform.
#         modified_waveform = np.fft.ifft(modified_fft_values)
#         return modified_waveform
# 
#     @staticmethod
#     def _get_calib_data(calibFilepath):
#         return np.loadtxt(calibFilepath, delimiter=",")
# 
#     @staticmethod
#     def get_calibration_func(calibration_data, dac_ref, freq_ref, attenuation):
#         from scipy.interpolate import CubicSpline
#         freq_MHz = calibration_data[0]
#         S21_dbm = calibration_data[1]
#         S21_interpolate = CubicSpline(freq_MHz, S21_dbm + attenuation)
#         interpolation = CubicSpline(freq_MHz, 10**(-(-S21_dbm + S21_interpolate(freq_ref))/10))
#         def calib_func(val):
#             f_min = np.min(freq_MHz)
#             f_max = np.max(freq_MHz)
#             val_array = np.atleast_1d(val)  # Ensure val is treated as a numpy array.
#             mask = (val_array >= f_min) & (val_array <= f_max)  # Create a mask for values within the range.
#             # For values within the calibrated range, use the interpolation function. Otherwise, return 1
#             result = np.where(mask, interpolation(val_array), 1)
#             if result.size == 1:
#                 return result.item()  # Return a scalar if the input was a scalar.
#             return result
# 
#         return calib_func
# 
#     @staticmethod
#     def get_calibration_func2(calibration_data, dac_ref, freq_ref, attenuation):
#         from scipy.interpolate import CubicSpline
#         freq_MHz = calibration_data[0]
#         S21_dbm = calibration_data[1]
#         S21_interpolate = CubicSpline(freq_MHz, S21_dbm + attenuation)
#         interpolation = CubicSpline(freq_MHz, 10 ** ((-S21_dbm + S21_interpolate(freq_ref)) / 10))
# 
#         def calib_func(val):
#             f_min = np.min(freq_MHz)
#             f_max = np.max(freq_MHz)
#             val_array = np.atleast_1d(val)  # Ensure val is treated as a numpy array.
#             mask = (val_array >= f_min) & (val_array <= f_max)  # Create a mask for values within the range.
#             # For values within the calibrated range, use the interpolation function. Otherwise, return 1
#             result = np.where(mask, interpolation(val_array), 1)
#             if result.size == 1:
#                 return result.item()  # Return a scalar if the input was a scalar.
#             return result
# 
#         return calib_func
# 
# 
# class DragModulation:
#     def __init__(self, drag_factor, drag_func: Callable = None):
#         """
#         Apply a drag modulation to the input waveform.
# 
#         Parameters:
#         - drag_factor: A coefficient for the drag correction amplitude
#         - drag_func: A callable accepting (waveform, drag_factor, sampling_rate) that returns the drag correction.
#         """
#         self.drag_factor = drag_factor
#         self.drag_func = drag_func if drag_func is not None else self._drag_func
# 
#     @staticmethod
#     def _drag_func(waveform, sampling_rate=1):
#         """drag correction for resonant driving"""
#         dt = 1/sampling_rate
#         return -1 * np.exp(1j * np.pi/2) * np.gradient(waveform, dt)
# 
#     def apply_modulation(self, waveform, sampling_rate=1):
#         wf_drag = self.drag_factor * self.drag_func(waveform, sampling_rate)
#         return waveform + wf_drag
# 
# 
# class ChirpModulation:
#     def __init__(self, chirp_func, maxf, maxv=None):
#         """
#         Apply a chirp modulation to the input waveform.
# 
#         Parameters:
#         - chirp_func: A callable accepting (amp, maxf, maxv) that returns the chirp frequency at the given amplitude.
#         - maxf: Maximum chirp instanteous frequency in MHz. Default value is 0
#         - maxv: Maximum amplitude of the given waveform. Default value is 30000
#         """
#         self.maxf = maxf
#         self.maxv = maxv if maxv is not None else 32766
#         self.chirp_func = chirp_func
# 
#     def _instant_frequency(self, waveform):
#         max_i = np.max(np.abs(np.real(waveform)))
#         max_q = np.max(np.abs(np.imag(waveform)))
#         # return self.chirp_func(np.abs(waveform), self.maxf, np.max((max_i, max_q)))
#         return self.chirp_func(np.abs(waveform), self.maxf, np.max(np.abs(waveform)))
#         # return self.chirp_func(np.abs(waveform), self.maxf, self.maxv)
# 
#     @staticmethod
#     def _chirp_phase(instant_freq, sampling_rate):
#         phase = np.zeros(len(instant_freq))
#         phi0 = 0
#         for i in range(len(instant_freq) - 1):
#             phase[i] = phi0
#             phi0 += np.pi * (instant_freq[i] + instant_freq[i + 1]) / sampling_rate
#         phase[-1] = phi0
#         return phase
# 
#     def apply_modulation(self, waveform, sampling_rate):
#         """
#         Apply a chirp modulation to the input waveform.
# 
#         Parameters:
#         - waveform: Input waveform array.
#         - sampling_rate: sampling_rate of the waveform
#         """
#         chirp_freq = self._instant_frequency(waveform)
#         chirp_phase = self._chirp_phase(chirp_freq, sampling_rate)
#         wf_chirp = waveform * np.exp(1j * chirp_phase)
# 
#         return wf_chirp
# 
# 
# class WaveformCorrection:
#     def __init__(self, filepath, freq, scale: str = "linear", max_scale=0.5):
#         self.calibration_data = self.get_calib_data(filepath)
#         self.freq = freq
#         self.scale = scale
#         self.max_scale = max_scale
#         if scale.lower() in ["db", "dbm", "log"]:
#             self.freq_ref = self.calibration_data[0][
#                 np.argmin(np.abs(self.calibration_data[1] - (np.max(self.calibration_data[1]) - 3)))]
#         elif scale.lower() == "linear":
#             self.freq_ref = self.calibration_data[0][
#                 np.argmin(np.abs(self.calibration_data[1] - max_scale*np.max(self.calibration_data[1])))]
# 
#         self.calibration_func = self.get_calibration_func(freq_ref=self.freq_ref, attenuation=0)
#         self.recover_func = self.get_recover_func(freq_ref=self.freq_ref, attenuation=0)
# 
#     @staticmethod
#     def get_calib_data(filepath):
#         data = np.loadtxt(filepath, delimiter=",")
#         return data
# 
#     @staticmethod
#     def compute_fourier_transform(signal, sampling_rate):
#         """
#         Compute the Fourier Transform of a signal.
# 
#         Parameters:
#         - signal: numpy array of the waveform data.
#         - sampling_rate: sampling rate in Hz.
# 
#         Returns:
#         - frequencies: numpy array of frequency bins (MHz).
#         - magnitudes: numpy array of corresponding magnitude spectrum.
#         """
#         N = len(signal)
#         fft_vals = np.fft.fft(signal)
#         fft_freq = np.fft.fftfreq(N, d=1 / sampling_rate)
# 
#         return fft_freq, fft_vals
# 
#     @staticmethod
#     def compute_inverse_fourier_transform(fft_signal):
#         """
#         Compute the Inverse Fourier Transform of a complex frequency-domain signal.
# 
#         Parameters:
#           fft_signal: numpy array of complex Fourier coefficients (full spectrum).
# 
#         Returns:
#           time_signal: numpy array representing the recovered complex time-domain signal.
#         """
#         time_signal = np.fft.ifft(fft_signal)
#         return time_signal
# 
#     def apply_modulation(self, waveform, sampling_rate):
#         """
#         Modify a waveform in the frequency domain and return the modified time-domain waveform.
# 
#         Parameters:
#             - signal (np.array): Input time-domain signal (can be complex or real).
#             - calibration_func (callable): A function that takes (fft_freq, fft_values) as inputs
#                                           and returns modified fft_values.
#             - sampling_rate (float): Sampling rate of the signal in Hz.
# 
#         Returns:
#             np.array: The modified time-domain signal (complex if the input was complex).
#         """
#         N = len(waveform)
#         t_list = np.arange(0, N) / sampling_rate
#         signal = waveform * np.exp(1j * 2 * np.pi * self.freq * t_list)
# 
#         # N = len(waveform)
#         # signal = waveform
# 
#         # Compute the Fourier Transform and associated frequency bins.
#         fft_values = np.fft.fft(signal)
#         fft_freq = np.fft.fftfreq(N, d=1 / sampling_rate)
# 
#         # limit pulse bandwidth by cutting off the component smaller than 0.01%
#         fft_max = np.max(np.abs(fft_values))
#         mask = (np.abs(fft_values)/fft_max >= 1e-4)
#         fft_values = np.where(mask, fft_values, 0)
# 
#         # Apply the provided modification function to the FFT coefficients.
#         modified_fft_values = self.calibration_func(fft_freq) * fft_values
# 
#         # Compute the inverse FFT to get back the modified time-domain waveform.
#         modified_signal = np.fft.ifft(modified_fft_values)
#         modified_waveform = modified_signal * np.exp(-1j * 2 * np.pi * self.freq * t_list)
# 
#         return modified_waveform
# 
#     def recover_modulation(self, waveform, sampling_rate):
#         """
#         Modify a waveform in the frequency domain and return the modified time-domain waveform.
# 
#         Parameters:
#             - signal (np.array): Input time-domain signal (can be complex or real).
#             - calibration_func (callable): A function that takes (fft_freq, fft_values) as inputs
#                                           and returns modified fft_values.
#             - sampling_rate (float): Sampling rate of the signal in Hz.
# 
#         Returns:
#             np.array: The modified time-domain signal (complex if the input was complex).
#         """
#         N = len(waveform)
#         t_list = np.arange(0, N) / sampling_rate
#         signal = waveform * np.exp(1j * 2 * np.pi * self.freq * t_list)
# 
#         # Compute the Fourier Transform and associated frequency bins.
#         fft_values = np.fft.fft(signal)
#         fft_freq = np.fft.fftfreq(N, d=1 / sampling_rate)
# 
#         # Apply the provided modification function to the FFT coefficients.
#         modified_fft_values = self.recover_func(fft_freq) * fft_values
# 
#         # Compute the inverse FFT to get back the modified time-domain waveform.
#         modified_signal = np.fft.ifft(modified_fft_values)
#         # if np.max(modified_signal) > 2**15:
#         #     modified_signal *= 2**15 / np.max(modified_signal)
#         #     warnings.warn("pulse amplitude exceeded maxv")
#         modified_waveform = modified_signal * np.exp(-1j * 2 * np.pi * self.freq * t_list)
#         return modified_waveform
# 
#     def get_calibration_func(self, freq_ref, attenuation):
#         from scipy.interpolate import CubicSpline
#         if self.scale.lower() in ["db", "dbm", "log"]:
#             freq_MHz = self.calibration_data[0]
#             S21_dbm = self.calibration_data[1]
#             S21_interpolate = CubicSpline(freq_MHz, S21_dbm + attenuation)
#             interpolation = CubicSpline(freq_MHz, 10 ** ((-S21_dbm + S21_interpolate(freq_ref)) / 10))
#         elif self.scale.lower() == "linear":
#             freq_MHz = self.calibration_data[0]
#             S21 = self.calibration_data[1]
#             S21_interpolate = CubicSpline(freq_MHz, S21 + attenuation)
#             interpolation = CubicSpline(freq_MHz, + S21_interpolate(freq_ref)/S21)
#         def calib_func(val):
#             f_min = np.min(freq_MHz)
#             f_max = np.max(freq_MHz)
#             val_array = np.atleast_1d(val)  # Ensure val is treated as a numpy array.
#             mask = (val_array >= f_min) & (val_array <= f_max)  # Create a mask for values within the range.
#             # For values within the calibrated range, use the interpolation function. Otherwise, return 1
#             result = np.where(mask, interpolation(val_array), 1)
#             if result.size == 1:
#                 return result.item()  # Return a scalar if the input was a scalar.
#             return result
#         return calib_func
# 
#     def get_recover_func(self, freq_ref, attenuation):
#         from scipy.interpolate import CubicSpline
#         if self.scale.lower() in ["db", "dbm"]:
#             freq_MHz = self.calibration_data[0]
#             S21_dbm = self.calibration_data[1]
#             S21_interpolate = CubicSpline(freq_MHz, S21_dbm + attenuation)
#             interpolation = CubicSpline(freq_MHz, 10 ** ((S21_dbm - S21_interpolate(freq_ref)) / 10))
#         elif self.scale.lower() == "linear":
#             freq_MHz = self.calibration_data[0]
#             S21 = self.calibration_data[1]
#             S21_interpolate = CubicSpline(freq_MHz, S21 + attenuation)
#             interpolation = CubicSpline(freq_MHz, + S21/S21_interpolate(freq_ref))
# 
#         def calib_func(val):
#             f_min = np.min(freq_MHz)
#             f_max = np.max(freq_MHz)
#             val_array = np.atleast_1d(val)  # Ensure val is treated as a numpy array.
#             mask = (val_array >= f_min) & (val_array <= f_max)  # Create a mask for values within the range.
#             result = np.where(mask, interpolation(val_array), 1)  # For values within the calibrated range, use the interpolation function. Otherwise, return 1
#             if result.size == 1:
#                 return result.item()  # Return a scalar if the input was a scalar.
#             return result
#         return calib_func
# 
#     def plot_calibration(self, ax=None):
#         if ax is None:
#             fig, ax = plt.subplots(1, 1)
#         else:
#             fig = ax.get_figure()
#         ax.set_title("calibration data")
#         ax.plot(self.calibration_data[0], self.calibration_data[1])
#         ax.set_xlabel("Frequency (MHz)")
#         return fig, ax
# 
# 
# class Gaussian(Waveform):
#     def __init__(self, prog, gen_ch, length, sigma, phase=0, maxv=None,
#                  padding: Union[float, List[float], None] = None):
#         super().__init__(prog, gen_ch, phase=phase, maxv=maxv)
#         self.sigma_samps = self.us_to_samps(sigma)
#         self.length_samps = self.us_to_samps(length)
#         self.padding = padding
#         self.waveform = self._generate_waveform(self.length_samps, self.sigma_samps)
# 
#     @staticmethod
#     def core(length, sigma):
#         """the definetion of Gaussian"""
#         t = np.arange(length)
#         y = np.exp(-(t - length / 2) ** 2 / sigma ** 2)
#         return y - np.min(y)
# 
#     def _generate_waveform(self, *args, **kwargs):
#         """
#         apply the necessary modificaiton to the core function,
#         generate in-phase (I) and quadrature (Q) components
#         """
#         waveform = self.maxv * self.core(*args, **kwargs)
#         waveform_padded = self._apply_padding(waveform, self.padding)
#         waveform_wphase = np.exp(1j * np.deg2rad(self.phase)) * waveform_padded
#         return waveform_wphase
# 
# 
# class TanhBox(Waveform):
#     def __init__(self, prog, gen_ch, length, ramp_width, cut_offset=0.01, phase=0, maxv=None,
#                  padding: Union[float, List[float], None] = None):
#         super().__init__(prog, gen_ch, phase=phase, maxv=maxv)
#         self.ramp_samps = self.us_to_samps(ramp_width)
#         self.length_samps = self.us_to_samps(length)
#         self.cut_offset = cut_offset
#         self.padding = padding
#         self.waveform = self._generate_waveform(self.length_samps, self.ramp_samps, self.cut_offset)
# 
#     @staticmethod
#     def core(length, ramp_width, cut_offset):
#         """
#         Create a numpy array containing a smooth box pulse made of two tanh functions subtract from each other.
# 
#         :param length: number of points of the pulse
#         :param ramp_width: number of points from cutOffset to 0.95 amplitude
#         :param cut_offset: the initial offset to cut on the tanh Function
#         :return:
#         """
#         t = np.arange(length)
#         c0_, c1_ = np.arctanh(2 * cut_offset - 1), np.arctanh(2 * 0.95 - 1)
#         k_ = (c1_ - c0_) / ramp_width
#         y = (0.5 * (np.tanh(k_ * t + c0_) - np.tanh(k_ * (t - length) - c0_)) - cut_offset) / (1 - cut_offset)
#         return y - np.min(y)
# 
#     def _generate_waveform(self, *args, **kwargs):
#         """
#         apply the necessary modificaiton to the core function,
#         generate in-phase (I) and quadrature (Q) components
#         """
#         waveform = self.maxv * self.core(*args, **kwargs)
#         waveform_padded = self._apply_padding(waveform, self.padding)
#         waveform_wphase = np.exp(1j * np.deg2rad(self.phase)) * waveform_padded
#         return waveform_wphase
# 
# 
# class FileDefined(Waveform, DragModulationMixin):
#     def __init__(self, prog, gen_ch, filepath, phase=0, maxv=None, drag_coeff=0,
#                  padding: Union[float, List[float], None] = None):
#         super().__init__(prog, gen_ch, phase=phase, maxv=maxv)
#         self.filepath = filepath
#         self.padding = padding
#         self.drag_coeff = drag_coeff
#         self.waveform = self._generate_waveform(filepath=self.filepath)
# 
#     @staticmethod
#     def core(filepath, **kwargs):
#         """
#         Reads waveform data (I, Q) from the file.
# 
#         param filepath: the filepath of the waveform
#         return:
#         """
#         # todo: deal with file formats
#         filetype = filepath.split(".")[-1]
#         if filetype == "npy":
#             data = np.load(filepath, **kwargs)
#             idata = data[:, 0]  # First column: I data
#             qdata = data[:, 1]  # Second column: Q data
#             return idata + 1j * qdata
#         elif filetype == "csv":
#             data = np.loadtxt(filepath, **kwargs)  # Assuming the file contains a two-column format (I, Q)
#             idata = data[:, 0]  # First column: I data
#             qdata = data[:, 1]  # Second column: Q data
#             return idata + 1j * qdata
#         else:
#             try:
#                 data = np.loadtxt(filepath, **kwargs)  # Assuming the file contains a two-column format (I, Q)
#                 idata = data[:, 0]  # First column: I data
#                 qdata = data[:, 1]  # Second column: Q data
#                 return idata + 1j * qdata
#             except Exception as e:
#                 raise ValueError(f"Error reading file {filepath}. Exception {e}")
#                 return -1
# 
#     def _generate_waveform(self, *args, **kwargs):
#         """
#         apply the necessary modificaiton to the core function,
#         generate in-phase (I) and quadrature (Q) components
#         """
#         waveform = self.core(*args, **kwargs)
#         waveform_padded = self._apply_padding(waveform, self.padding)
#         waveform_wphase = np.exp(1j * np.deg2rad(self.phase)) * waveform_padded
#         waveform_dragged = self.apply_drag_modulation(waveform_wphase, drag_coeff=self.drag_coeff)
#         return waveform_dragged
# 
# 
# class GaussianModulated(Waveform):
#     def __init__(self, prog, gen_ch, length, sigma, phase=0, maxv=None, modulations: list = (),
#                  padding: Union[float, List[float], None] = None, shape=None):
#         super().__init__(prog, gen_ch, phase=phase, maxv=maxv)
#         self.sigma_samps = self.us_to_samps(sigma)
#         self.length_samps = self.us_to_samps(length)
#         self.padding = padding
#         self.modulations = modulations
#         self.waveform = self._generate_waveform(self.length_samps, self.sigma_samps)
#         shape = shape if shape is not None else self.__class__.__name__
#         WaveformRegistry.register(shape, self.__class__)
# 
#     @staticmethod
#     def core(length, sigma):
#         """the definetion of Gaussian"""
#         t = np.arange(length)
#         y = np.exp(-(t - length / 2) ** 2 / sigma ** 2)
#         return y - np.min(y)
# 
#     def _generate_waveform(self, *args, **kwargs):
#         """
#         apply the necessary modificaiton to the core function,
#         generate in-phase (I) and quadrature (Q) components
#         """
#         waveform = self.maxv * self.core(*args, **kwargs)
#         waveform = self._apply_padding(waveform, self.padding)
#         waveform = np.exp(1j * np.deg2rad(self.phase)) * waveform
#         for mod in self.modulations:
#             waveform = mod.apply_modulation(waveform, self.sampling_rate)
#         return waveform
# 
# 
# class TanhBoxModulated(Waveform):
#     def __init__(self, prog, gen_ch, length, ramp_width, cut_offset=0.01, phase=0, maxv=None, modulations: list = (),
#                  padding: Union[float, List[float], None] = None, shape=None):
#         super().__init__(prog, gen_ch, phase=phase, maxv=maxv)
#         self.ramp_samps = self.us_to_samps(ramp_width)
#         self.length_samps = self.us_to_samps(length)
#         self.cut_offset = cut_offset
#         self.padding = padding
#         self.modulations = modulations
#         self.waveform = self._generate_waveform(self.length_samps, self.ramp_samps, self.cut_offset)
#         shape = shape if shape is not None else self.__class__.__name__
#         WaveformRegistry.register(shape, self.__class__)
# 
#     @staticmethod
#     def core(length, ramp_width, cut_offset):
#         """
#         Create a numpy array containing a smooth box pulse made of two tanh functions subtract from each other.
# 
#         :param length: number of points of the pulse
#         :param ramp_width: number of points from cutOffset to 0.95 amplitude
#         :param cut_offset: the initial offset to cut on the tanh Function
#         :return:
#         """
#         t = np.arange(length)
#         c0_, c1_ = np.arctanh(2 * cut_offset - 1), np.arctanh(2 * 0.95 - 1)
#         k_ = (c1_ - c0_) / ramp_width
#         y = (0.5 * (np.tanh(k_ * t + c0_) - np.tanh(k_ * (t - length) - c0_)) - cut_offset) / (1 - cut_offset)
#         return y - np.min(y)
# 
#     def _generate_waveform(self, *args, **kwargs):
#         """
#         apply the necessary modificaiton to the core function,
#         generate in-phase (I) and quadrature (Q) components
#         """
#         waveform = self.maxv * self.core(*args, **kwargs)
#         waveform = self._apply_padding(waveform, self.padding)
#         waveform = np.exp(1j * np.deg2rad(self.phase)) * waveform
#         for mod in self.modulations:
#             waveform = mod.apply_modulation(waveform, self.sampling_rate)
#         return waveform
# 
# 
# class ConcatenateWaveform(Waveform):
#     def __init__(self, prog, gen_ch, waveforms: List[Waveform], phase=0, maxv=None, shape=None):
#         super().__init__(prog, gen_ch, phase, maxv)
#         self.wavefrom_list = waveforms
#         self.waveform = self._generate_waveform()
#         shape = shape if shape is not None else self.__class__.__name__
#         WaveformRegistry.register(shape, self.__class__)
# 
#     def _generate_waveform(self):
#         return np.concatenate([w.waveform for w in self.wavefrom_list])
# 
# 
# def add_waveform(prog: QickProgram, gen_ch, name, shape, **kwargs):
#     """Adds a waveform to the DAC channel, using physical parameters of the pulse.
#     The pulse will peak at length/2.
# 
#     Parameters
#     ----------
#     prog: QickProgram
#         The experiment QickProgram
#     gen_ch : str
#         name of the generator channel
#     name : str
#         Name of the pulse
#     shape : str
#         shape/type of the pulse, e.g. Gaussian, TanhBoxChirp
#     """
#     if shape.lower() in (wave.lower() for wave in WaveformRegistry.available_waveforms()):
#         pulse = WaveformRegistry.create(shape=shape, prog=prog, gen_ch=gen_ch, **kwargs)
#         # pulse.plot_waveform()
#         pulse.add_waveform(prog, name=name)
#     else:
#         raise NameError(f"Unsupported pulse shape {shape}."
#                         f"Choose from available shapes: {WaveformRegistry.available_waveforms()},"
#                         f"or define new waveforms.")
# 
# 
# def add_waveform_concatenate(prog: QickProgram, gen_ch: str | int, name, gatelist, maxv=None):
#     pass


if __name__ == "__main__":
    pass

