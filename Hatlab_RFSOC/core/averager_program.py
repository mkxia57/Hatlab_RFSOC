from typing import Dict, List, Union, Callable, Literal, Tuple

from tqdm import tqdm
import numpy as np
from qick.qick_asm import QickProgram

from .pulses import add_gaussian, add_tanh

RegisterTypes = Literal["freq", "time", "phase", "adc_freq"]


class QickRegister:
    def __init__(self, prog:QickProgram, page: int, addr: int, reg_type: RegisterTypes = None,
                 gen_ch: int = None, ro_ch: int = None, init_val=None, name: str = None):
        """
        keeps the generator/readout channel and register type information, for automatically using them when converting
        value to register.

        :param prog: Qick program in which the register is used.
        :param page: page of the register
        :param addr: address of the register in the register page (the register number)
        :param reg_type: type of the register, used for automatic conversion to values.
        :param gen_ch: generator channel numer to which the register is associated with.
        :param ro_ch: readout channel numer to which the register is associated with.
        :param init_val: initial value of the register. If reg_type is not None, the value should be in its physical unit.
        :param name: name of the register
        """
        self.prog = prog
        self.page = page
        self.addr = addr
        self.type = reg_type
        self.gen_ch = gen_ch
        self.ro_ch = ro_ch
        self.init_val = init_val
        self.name = name
        if init_val is not None:
            self.reset()

    def val2reg(self, val):
        """
        convert physical value to a qick register value
        :param val:
        :return:
        """
        if self.type == "freq":
            return self.prog.freq2reg(val, self.gen_ch, self.ro_ch)
        elif self.type == "time":
            if self.gen_ch is not None:
                return self.prog.us2cycles(val, self.gen_ch)
            else:
                return self.prog.us2cycles(val, self.gen_ch, self.ro_ch)
        elif self.type == "phase":
            return self.prog.deg2reg(val, self.gen_ch)
        elif self.type == "adc_freq":
            return self.prog.freq2reg_adc(val, self.ro_ch, self.gen_ch)
        else:
            return np.int32(val)

    def reg2val(self, reg):
        """
        converts a qick register value to its value in physical units
        :param reg:
        :return:
        """
        if self.type == "freq":
            return self.prog.reg2freq(reg, self.gen_ch)
        elif self.type == "time":
            if self.gen_ch is not None:
                return self.prog.cycles2us(reg, self.gen_ch)
            else:
                return self.prog.cycles2us(reg, self.gen_ch, self.ro_ch)
        elif self.type == "phase":
            return self.prog.reg2deg(reg, self.gen_ch)
        elif self.type == "adc_freq":
            return self.prog.reg2freq_adc(reg, self.ro_ch)
        else:
            return reg

    def reset(self):
        """
        reset register value to its init_val
        :return:
        """
        self.prog.safe_regwi(self.page, self.addr, self.val2reg(self.init_val))


class QickSweep:
    """
    QickSweep class, describes a sweeps over a qick register.
    """

    def __init__(self, prog: QickProgram, reg: QickRegister, start, stop, expts: int):
        """

        :param prog: QickProgram in which the sweep happens.
        :param reg: QickRegister object associated to the register to sweep.
        :param start: start value of the register to sweep, in physical units
        :param stop: stop value of the register to sweep, in physical units
        :param expts: number of experiment points between start and stop value.
        """
        self.prog = prog
        self.reg = reg
        self.start = start
        self.stop = stop
        self.expts = expts
        step_val = (stop - start) / (expts - 1)
        self.reg_step = reg.val2reg(step_val)
        self.init_val = start
        self.sweep_array = np.linspace(start, stop, expts)

    def update(self):
        """
        update the register value. This will be called after finishing last register sweep.
        This function should be overwritten if more complicated update is needed.
        :return:
        """
        self.prog.mathi(self.reg.page, self.reg.addr, self.reg.addr, '+', self.reg_step)

    def reset(self):
        """
        reset the register to the start value. will be called at the beginning of each sweep.
        This function should be overwritten if more complicated reset is needed.
        :return:
        """
        self.reg.reset()


class APAveragerProgram(QickProgram):
    """
    APAveragerProgram class. "A" and "P" stands for "Automatic" and "Physical". This class automatically declares the
    generator and readout channels using the parameters provided in cfg["gen_chs"] and cfg["ro_chs"], and contains
    functions that hopefully can make it easier to program pulse sequences with parameters in their physical units
    (so that we don't have to constantly call "_2reg"/"_2cycles" functions).
    The "acquire" methods are copied from qick.RAveragerProgram.

    config requirements:
    "gen_chs" = dictionary that contains the configuration of each generator channel;
        The format should be: {"gen_name": {**kwargs_of_declare_gen}}
    "ro_chs" = dictionary that contains the configuration of each readout channel;
        The format should be: {"ro_name": {**kwargs_of_declare_readout}}
    "waveforms"(optional) =  dictionary that contains some waveforms and their parameters (in physical units)
        The format should be: {"waveform_name": {**kwargs_of_add_waveform}}
    """

    def __init__(self, soccfg, cfg):
        """
        Initialize the QickProgram and automatically declares all generator and readout channels in cfg["gen_chs"] and
        cfg["ro_chs"]
        """
        super().__init__(soccfg)
        self.cfg = cfg
        self.user_reg_dict = {}  # look up dict for registers defined in each generator channel
        self._user_regs = []  # (page, addr) of all user defined registers
        self.expts = None  # abstract variable for total number of experiments in each repetition.
        self.declare_all_gens()
        self.declare_all_readouts()

    def declare_all_gens(self):
        """
        Declare all generators in the config dict based on the items specified in cfg["gen_chs"].

        :return:
        """
        for gen_ch, kws in self.cfg["gen_chs"].items():
            if ("ch_I" in kws) and ("ch_Q" in kws):
                ch_i = kws.pop("ch_I")
                ch_q = kws.pop("ch_Q")
                for arg in ["skew_phase", "IQ_scale"]:
                    try:
                        kws.pop(arg)
                    except AttributeError:
                        pass
                self.declare_gen(ch_i, **kws)
                self.declare_gen(ch_q, **kws)  # todo: all the other functions doesn't support IQ channel gen yet...
            else:
                self.declare_gen(**kws)
            self.user_reg_dict[gen_ch] = {}

    def declare_all_readouts(self):
        """
        Declare all readout channels in the config dict based on the items specified in cfg["ro_chs"].

        :return:
        """
        for ro_ch, kws in self.cfg["ro_chs"].items():
            self.declare_readout(**kws)

    def get_reg(self, gen_ch: str, name: str) -> QickRegister:
        """
        Gets tProc register page and address associated with gen_ch and register name. Creates a QickRegister object for
        return.

        :param gen_ch: name of the generator channel, as in cfg["gen_chs"]
        :param name:  name of the qick register, as in QickProgram.pulse_registers
        :return: QickRegister
        """
        gen_cgf = self.cfg["gen_chs"][gen_ch]
        page = self.ch_page(gen_cgf["ch"])
        addr = self.sreg(gen_cgf["ch"], name)
        reg_type = name if name in RegisterTypes.__args__ else None
        reg = QickRegister(self, page, addr, reg_type, gen_cgf["ch"], gen_cgf.get("ro_ch"), name=f"{gen_ch}_{name}")
        return reg

    def new_reg(self, gen_ch: str, name: str = None, init_val=None, reg_type: RegisterTypes = None) -> QickRegister:
        """
        Declare a new register in the generator register page. Address automatically adds 1 one when each time a new
        register in the same page is declared.

        :param gen_ch: name of the generator channel, as in cfg["gen_chs"]
        :param name: name of the new register. Optional.
        :param init_val: initial value for the register, when reg_type is provided, the reg_val should be in the unit of
            the corresponding type.
        :param reg_type: type of the register, e.g. freq, time, phase.
        :return: QickRegister
        """
        gen_cgf = self.cfg["gen_chs"][gen_ch]
        page = self.ch_page(gen_cgf["ch"])
        addr = 1
        while (page, addr) in self._user_regs:
            addr += 1
        if addr > 12:
            raise ValueError(f"registers in page {page} ({gen_ch}) is full.")
        self._user_regs.append((page, addr))

        if name is None:
            name = f"reg_{addr}"
        if name in self.user_reg_dict[gen_ch].keys():
            raise KeyError(f"register name '{name}' already exists for channel {gen_ch}")

        reg = QickRegister(self, page, addr, reg_type, gen_cgf["ch"], gen_cgf.get("ro_ch"), init_val, name=name)
        self.user_reg_dict[gen_ch][name] = reg

        return reg

    def set_pulse_params(self, gen_ch: str, **kwargs):
        """
        This is a wrapper of the QickProgram.set_pulse_registers. Instead of taking register values, this function takes
        the physical values of the pulse parameters. E.g. freq in MHz, length in us, phase in degree.

        Parameters
        ----------
        gen_ch : str
            name of the DAC channel
        style : str
            Pulse style ("const", "arb", "flat_top")
        freq : float
            Frequency (MHz)
        phase : float
            Phase (deg)
        gain : int
            Gain (DAC units)
        phrst : int
            If 1, it resets the phase coherent accumulator
        stdysel : str
            Selects what value is output continuously by the signal generator after the generation of a pulse. If "last", it is the last calculated sample of the pulse. If "zero", it is a zero value.
        mode : str
            Selects whether the output is "oneshot" or "periodic"
        outsel : str
            Selects the output source. The output is complex. Tables define envelopes for I and Q. If "product", the output is the product of table and DDS. If "dds", the output is the DDS only. If "input", the output is from the table for the real part, and zeros for the imaginary part. If "zero", the output is always zero.
        length : float
            length of the constant pulse or the flat part of the flat_top pulse, in us
        waveform : str
            Name of the envelope waveform loaded with add_pulse(), used for "arb" and "flat_top" styles
        mask : list of int
            for a muxed signal generator, the list of tones to enable for this pulse
        """
        kw_reg = kwargs.copy()
        gen_cgf = self.cfg["gen_chs"][gen_ch]
        if "freq" in kwargs:
            kw_reg["freq"] = self.soccfg.freq2reg(kwargs["freq"], gen_cgf["ch"], gen_cgf.get("ro_ch"))
        if "phase" in kwargs:
            kw_reg["phase"] = self.soccfg.deg2reg(kwargs["phase"], gen_cgf["ch"])
        if "length" in kwargs:
            kw_reg["length"] = self.soccfg.us2cycles(kwargs["length"], gen_cgf["ch"])
        self.set_pulse_registers(gen_cgf["ch"], **kw_reg)

    def add_waveform(self, gen_ch, name, shape, **kwargs):
        """
        Add waveform to a generator channel based on parameters specified in cfg["waveforms"]

        :param gen_ch: name of the generator channel, as in cfg["gen_chs"]
        :param name: name of the waveform.
        :param shape: shape of the waveform, should be one of the waveforms that are available in pulses.py
        :param kwargs: kwargs for the pulse

        :return:
        """

        # todo: the parser can be better... instead of using if commands to select from only two possible waveforms here
        #   We should have a abstract waveform class, each new waveform should be written as a waveform class instance,
        #   and the parser should search for waveform in pulses.py (rename that to waveforms.py)

        if shape == "gaussian":
            add_gaussian(self, gen_ch, name, **kwargs)
        elif shape == "tanh_box":
            add_tanh(self, gen_ch, name, **kwargs)
        else:
            raise NameError(f"unsupported pulse shape {shape}")

    def add_waveform_from_cfg(self, gen_ch: str, name: str):
        """
        Add waveform to a generator channel based on parameters specified in cfg["waveforms"]

        :param gen_ch: name of the generator channel, as in cfg["gen_chs"]
        :param name: name of the waveform, as in cfg["waveforms"]
        :return:
        """
        pulse_params = self.cfg["waveforms"][name]
        self.add_waveform(gen_ch, name, **pulse_params)

    def get_expt_pts(self):
        """
        Abstract method for calculating total experiment points in each repetition based on the config.
        """
        pass

    def acquire_round(self, soc, threshold=None, angle=None, readouts_per_experiment=1, save_experiments=None,
                      load_pulses=True, start_src="internal", progress=False, debug=False):
        """
        This method optionally loads pulses on to the SoC, configures the ADC readouts, loads the machine code representation of the AveragerProgram onto the SoC, starts the program and streams the data into the Python, returning it as a set of numpy arrays.

        config requirements:
        "reps" = number of repetitions;

        :param soc: Qick object
        :type soc: Qick object
        :param threshold: threshold
        :type threshold: int
        :param angle: rotation angle
        :type angle: list
        :param readouts_per_experiment: readouts per experiment
        :type readouts_per_experiment: int
        :param save_experiments: saved experiments
        :type save_experiments: list
        :param load_pulses: If true, loads pulses into the tProc
        :type load_pulses: bool
        :param start_src: "internal" (tProc starts immediately) or "external" (waits for an external trigger)
        :type start_src: string
        :param progress: If true, displays progress bar
        :type progress: bool
        :param debug: If true, displays assembly code for tProc program
        :type debug: bool
        :returns:
            - avg_di (:py:class:`list`) - list of lists of averaged accumulated I data for ADCs 0 and 1
            - avg_dq (:py:class:`list`) - list of lists of averaged accumulated Q data for ADCs 0 and 1
        """

        if angle is None:
            angle = [0, 0]
        if save_experiments is None:
            save_experiments = [0]
        if load_pulses:
            self.load_pulses(soc)

        # Configure signal generators
        self.config_gens(soc)

        # Configure the readout down converters
        self.config_readouts(soc)
        self.config_bufs(soc, enable_avg=True, enable_buf=True)

        # load this program into the soc's tproc
        self.load_program(soc, debug=debug)

        # configure tproc for internal/external start
        soc.start_src(start_src)

        reps, expts = self.cfg['reps'], self.expts

        count = 0
        total_count = reps * expts * readouts_per_experiment
        n_ro = len(self.ro_chs)

        d_buf = np.zeros((n_ro, 2, total_count))
        self.stats = []

        with tqdm(total=total_count, disable=not progress) as pbar:
            soc.start_readout(total_count, counter_addr=1, ch_list=list(
                self.ro_chs), reads_per_count=readouts_per_experiment)
            while count < total_count:
                new_data = soc.poll_data()
                for d, s in new_data:
                    new_points = d.shape[2]
                    d_buf[:, :, count:count + new_points] = d
                    count += new_points
                    self.stats.append(s)
                    pbar.update(new_points)

        # reformat the data into separate I and Q arrays
        di_buf = d_buf[:, 0, :]
        dq_buf = d_buf[:, 1, :]

        # save results to class in case you want to look at it later or for analysis
        self.di_buf = di_buf
        self.dq_buf = dq_buf

        if threshold is not None:
            self.shots = self.get_single_shots(
                di_buf, dq_buf, threshold, angle)

        expt_pts = self.get_expt_pts()

        avg_di = np.zeros((n_ro, len(save_experiments), expts))
        avg_dq = np.zeros((n_ro, len(save_experiments), expts))

        for nn, ii in enumerate(save_experiments):
            for i_ch, (ch, ro) in enumerate(self.ro_chs.items()):
                if threshold is None:
                    avg_di[i_ch][nn] = np.sum(di_buf[i_ch][ii::readouts_per_experiment].reshape(
                        (reps, expts)), 0) / (reps) / ro.length
                    avg_dq[i_ch][nn] = np.sum(dq_buf[i_ch][ii::readouts_per_experiment].reshape(
                        (reps, expts)), 0) / (reps) / ro.length
                else:
                    avg_di[i_ch][nn] = np.sum(
                        self.shots[i_ch][ii::readouts_per_experiment].reshape((reps, expts)), 0) / (reps)
                    avg_dq = np.zeros(avg_di.shape)

        return expt_pts, avg_di, avg_dq

    def get_single_shots(self, di, dq, threshold, angle=None):
        """
        This method converts the raw I/Q data to single shots according to the threshold and rotation angle

        :param di: Raw I data
        :type di: list
        :param dq: Raw Q data
        :type dq: list
        :param threshold: threshold
        :type threshold: int
        :param angle: rotation angle
        :type angle: list

        :returns:
            - single_shot_array (:py:class:`array`) - Numpy array of single shot data

        """

        if angle is None:
            angle = [0, 0]
        if type(threshold) is int:
            threshold = [threshold, threshold]
        return np.array([np.heaviside(
            (di[i] * np.cos(angle[i]) - dq[i] * np.sin(angle[i])) / self.ro_chs[ch].length - threshold[i], 0) for i, ch
            in enumerate(self.ro_chs)])

    def acquire(self, soc, threshold=None, angle=None, load_pulses=True, readouts_per_experiment=1,
                save_experiments=None, start_src="internal", progress=False, debug=False):
        """
        This method optionally loads pulses on to the SoC, configures the ADC readouts, loads the machine code representation of the AveragerProgram onto the SoC, starts the program and streams the data into the Python, returning it as a set of numpy arrays.
        config requirements:
        "reps" = number of repetitions;

        :param soc: Qick object
        :type soc: Qick object
        :param threshold: threshold
        :type threshold: int
        :param angle: rotation angle
        :type angle: list
        :param readouts_per_experiment: readouts per experiment
        :type readouts_per_experiment: int
        :param save_experiments: saved experiments
        :type save_experiments: list
        :param load_pulses: If true, loads pulses into the tProc
        :type load_pulses: bool
        :param start_src: "internal" (tProc starts immediately) or "external" (each round waits for an external trigger)
        :type start_src: string
        :param progress: If true, displays progress bar
        :type progress: bool
        :param debug: If true, displays assembly code for tProc program
        :type debug: bool
        :returns:
            - expt_pts (:py:class:`list`) - list of experiment points
            - avg_di (:py:class:`list`) - list of lists of averaged accumulated I data for ADCs 0 and 1
            - avg_dq (:py:class:`list`) - list of lists of averaged accumulated Q data for ADCs 0 and 1
        """
        reps, expts, rounds = self.cfg['reps'], self.expts, self.cfg.get("rounds", 1)
        msmt_per_rep = expts * readouts_per_experiment
        tot_reps = reps * rounds
        total_msmt = msmt_per_rep * tot_reps

        n_ro = len(self.ro_chs)

        self.di_buf_p = np.zeros((n_ro, tot_reps, msmt_per_rep))
        self.dq_buf_p = np.zeros((n_ro, tot_reps, msmt_per_rep))

        if angle is None:
            angle = [0, 0]
        if save_experiments is None:
            save_experiments = [0]
        if "rounds" not in self.cfg or self.cfg["rounds"] == 1:
            expt_pts, avg_di, avg_dq = self.acquire_round(soc, threshold=threshold, angle=angle,
                                                          readouts_per_experiment=readouts_per_experiment,
                                                          save_experiments=save_experiments, load_pulses=load_pulses,
                                                          start_src=start_src, progress=progress, debug=debug)
            self.di_buf_p = self.di_buf.reshape(n_ro, reps, -1)
            self.dq_buf_p = self.dq_buf.reshape(n_ro, reps, -1)
            return expt_pts, avg_di, avg_dq

        avg_di = None
        for ii in tqdm(range(rounds), disable=not progress):
            expt_pts, avg_di0, avg_dq0 = self.acquire_round(soc, threshold=threshold, angle=angle,
                                                            readouts_per_experiment=readouts_per_experiment,
                                                            save_experiments=save_experiments, load_pulses=load_pulses,
                                                            start_src=start_src, progress=progress, debug=debug)

            if avg_di is None:
                avg_di, avg_dq = avg_di0, avg_dq0
            else:
                avg_di += avg_di0
                avg_dq += avg_dq0

            self.di_buf_p[:, reps * ii: reps * (ii + 1), :] = self.di_buf.reshape(n_ro, reps, -1)
            self.dq_buf_p[:, reps * ii: reps * (ii + 1), :] = self.dq_buf.reshape(n_ro, reps, -1)

        return expt_pts, avg_di / self.cfg["rounds"], avg_dq / self.cfg["rounds"]


class NDAveragerProgram(APAveragerProgram):
    """
    NDAveragerProgram class, for qubit experiments that sweep over multiple variables in qick.

    :param cfg: Configuration dictionary
    :type cfg: dict
    """

    def __init__(self, soccfg, cfg):
        """
        Constructor for the NDAveragerProgram. Make the ND sweep asm commands.
        """
        super().__init__(soccfg, cfg)
        self.qick_sweeps: List[QickSweep] = []
        self.expts = 1
        self.make_program()

    def initialize(self):
        """
        Abstract method for initializing the program and can include any instructions that are executed once at the beginning of the program.
        """
        pass

    def body(self):
        """
        Abstract method for the body of the program
        """
        pass

    def add_sweep(self, sweep: QickSweep):
        """
        Add a sweep to the qick asm program. The order of sweeping will follow first added first sweep.
        :param sweep:
        :return:
        """
        self.qick_sweeps.append(sweep)
        self.expts *= sweep.expts

    def make_program(self):
        """
        Make the N dimensional sweep program. The program will run initialize once at the beginning, then iterate over
        all the sweep parameters and run the body. The whole sweep will repeat for cfg["reps"] number of times.
        """
        p = self

        p.initialize()  # initialize only run once at the very beginning

        rcount = 13  # total run counter
        rep_count = 14  # repetition counter

        n_sweeps = len(self.qick_sweeps)
        counter_regs = (
                np.arange(n_sweeps) + 15).tolist()  # not sure why this has to be a list (np.array doesn't work)...
        if counter_regs[-1] > 21:
            raise OverflowError(f"too many qick inner loops ({n_sweeps}), run out of counter registers")

        p.regwi(0, rcount, 0)  # reset total run count

        # set repetition counter and tag
        p.regwi(0, rep_count, self.cfg["reps"] - 1)
        p.label("LOOP_rep")

        # add reset and staring tags for each sweep
        for creg, swp in zip(counter_regs[::-1], self.qick_sweeps[::-1]):
            swp.reset()
            p.regwi(0, creg, swp.expts - 1)
            p.label(f"LOOP_{swp.reg.name if swp.reg.name is not None else creg}")

        # run body and total_run_counter++
        p.body()
        p.mathi(0, rcount, rcount, "+", 1)
        p.memwi(0, rcount, 1)

        # add update and stop condition for each sweep
        for creg, swp in zip(counter_regs, self.qick_sweeps):
            swp.update()
            p.loopnz(0, creg, f"LOOP_{swp.reg.name if swp.reg.name is not None else creg}")

        # stop condition for repetition
        p.loopnz(0, rep_count, 'LOOP_rep')

        p.end()

    def get_expt_pts(self):
        """
        :return:
        """
        sweep_pts = []
        for swp in self.qick_sweeps:
            sweep_pts.append(swp.sweep_array)
        return sweep_pts