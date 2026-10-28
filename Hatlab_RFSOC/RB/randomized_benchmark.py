from typing import Dict, List, Union, Callable
import random

import numpy as np
import qutip as qt

try:
    # QuTiP 5 + qutip-qip
    from qutip_qip.operations import (
        rx, ry, rz, iswap, cnot, swap, sqrtiswap,
    )
except ImportError:
    # QuTiP 4.x
    from qutip.qip.operations import (
        rx, ry, rz, iswap, cnot, swap, sqrtiswap,
    )

from qick.averager_program import AveragerProgram
from qick.asm_v1 import QickProgram
from Hatlab_RFSOC.core.averager_program import QCAveragerProgram, QickSweep, QubitMsmtMixin
from Hatlab_RFSOC.core.circuit_program import (
    CircuitProgram, GateLibrary, PhaseCorrection, NullPhaseCorrection, SubharmonicPhaseCorrection,
)

PI = np.pi

qubitGateBasis = {
    'x': rx(PI),
    'x2': rx(PI / 2),
    'x2N': rx(-PI / 2),
    'y': ry(PI),
    'y2': ry(PI / 2),
    'y2N': ry(-PI / 2),
    'z': rz(PI),
    'z2': rz(PI / 2),
    'z2N': rz(-PI / 2),
    'Id': qt.identity(2),
    'iSWAP': iswap(),
    'CNOT': cnot(),
    'SWAP': swap(),
    'rtiSW': sqrtiswap()
}


class CliffordUtils:
    # cliffordGroup = {
    #     'c_1': ['Id'],
    #     'c_2': ['x2', 'y2', 'x2N'],
    #     'c_3': ['x', 'y'],
    #     'c_4': ['x2', 'y2N', 'x2N'],  # identity
    #     'c_5': ['y'],
    #     'c_6': ['x2', 'y2', 'x2'],
    #     'c_7': ['x'],
    #     'c_8': ['x2', 'y2N', 'x2'],  # pi-pulse
    #     'c_9': ['x', 'y2'],
    #     'c_10': ['y2N'],
    #     'c_11': ['y2N', 'x2'],
    #     'c_12': ['y2N', 'x2N'],  # pi/2 along negative-y
    #     'c_13': ['x', 'y2N'],
    #     'c_14': ['y2', 'x2N'],
    #     'c_15': ['y2'],
    #     'c_16': ['y2', 'x2'],  # pi/2 along positve-y
    #     'c_17': ['x2N', 'y2N'],
    #     'c_18': ['x2N', 'y2'],
    #     'c_19': ['x2N', 'y'],
    #     'c_20': ['x2N'],  # pi/2 along negative-x
    #     'c_21': ['x2', 'y2N'],
    #     'c_22': ['x2'],
    #     'c_23': ['x2', 'y'],
    #     'c_24': ['x2', 'y2']  # pi/2 along postive-x
    # }

    cliffordGroup = {
        'c_1': ['Id'],
        'c_2': ['x'],
        'c_3': ['x2'],
        'c_4': ['x2N'],
        'c_5': ['y'],
        'c_6': ['y2'],
        'c_7': ['y2N'],
        'c_8': ['x', 'y'],  # z
        'c_9': ['x2', 'y2', 'x2N'],  # z2
        'c_10': ['x2', 'y2N', 'x2N'],  # z2n
        'c_11': ['x2', 'y2', 'x2'],  # x+y pi
        'c_12': ['x2', 'y2N', 'x2'],  # x-y pi
        'c_13': ['x', 'y2'],  # x-z pi
        'c_14': ['x', 'y2N'],  # x+z pi
        'c_15': ['x2N', 'y'],  # y-z pi
        'c_16': ['x2', 'y'],  # y+z pi
        'c_17': ['y2N', 'x2'],  # -z-y+x
        'c_18': ['y2N', 'x2N'],  # -z+y+x
        'c_19': ['y2', 'x2N'],  # z-y+x
        'c_20': ['y2', 'x2'],  # z+y+x
        'c_21': ['x2N', 'y2N'],  # y+z+x
        'c_22': ['x2N', 'y2'],  # -y-z+x
        'c_23': ['x2', 'y2N'],  # -y+z+x
        'c_24': ['x2', 'y2']  # y-z-x
    }

    # cliffordGroupZ = {
    #     'c_1': ['Id'],
    #     'c_2': ['z2'],
    #     'c_3': ['z'],
    #     'c_4': ['z2N'],  # identity
    #     'c_5': ['y'],
    #     'c_6': ['x', 'z2N'],
    #     'c_7': ['x'],
    #     'c_8': ['x', 'z2'],  # pi-pulse
    #     'c_9': ['z', 'y2N'],
    #     'c_10': ['y2N'],
    #     'c_11': ['y2N', 'x2'],
    #     'c_12': ['y2N', 'x2N'],  # pi/2 along negative-y
    #     'c_13': ['z', 'y2'],
    #     'c_14': ['y2', 'x2N'],
    #     'c_15': ['y2'],
    #     'c_16': ['y2', 'x2'],  # pi/2 along positve-y
    #     'c_17': ['x2N', 'y2N'],
    #     'c_18': ['x2N', 'y2'],
    #     'c_19': ['x2', 'z'],
    #     'c_20': ['x2N'],  # pi/2 along negative-x
    #     'c_21': ['x2', 'y2N'],
    #     'c_22': ['x2'],
    #     'c_23': ['x2N', 'z'],
    #     'c_24': ['x2', 'y2']  # pi/2 along postive-x
    # }

    cliffordGroupZ = {
        'c_1': ['Id'],
        'c_2': ['x'],
        'c_3': ['x2'],
        'c_4': ['x2N'],
        'c_5': ['y'],
        'c_6': ['y2'],
        'c_7': ['y2N'],
        'c_8': ['z'],
        'c_10': ['z2'],
        'c_9': ['z2N'],
        'c_12': ['x', 'z2N'],
        'c_11': ['x', 'z2'],
        'c_13': ['z', 'y2N'],
        'c_14': ['z', 'y2'],
        'c_15': ['x2', 'z'],
        # 'c_14': ['x', 'y2N'],  # x+z pi
        # 'c_15': ['x2N', 'y'],  # y-z pi
        'c_16': ['x2N', 'z'],
        'c_17': ['y2N', 'x2'],
        'c_18': ['y2N', 'x2N'],
        'c_19': ['y2', 'x2N'],
        'c_20': ['y2', 'x2'],
        'c_21': ['x2N', 'y2N'],
        'c_22': ['x2N', 'y2'],
        'c_23': ['x2', 'y2N'],
        'c_24': ['x2', 'y2']
    }

    sGroup = {
        's_0': ['Id'],
        's_1': ['z2', 'x2'],
        's_2': ['z', 'x2', 'z2']
    }

    twoQubitGroup = {
        'iSWAP': ['iSWAP'],
        'CNOT': [[['z2N'], ['x2', 'z2']], 'iSWAP', [['x2'], ['Id']], 'iSWAP', [['Id'], ['z2']]],
        'SWAP': ['iSWAP', [['Id'], ['x2N']], 'iSWAP', [['x2N'], ['Id']], 'iSWAP', [['Id'], ['x2N']]]
    }

    twoQubitGroupRt = {
        'iSWAP': ['rtiSW', 'rtiSW'],
        'CNOT': [[['x2', 'z'], ['x', 'z2N']], 'rtiSW', [['x'], ['z']], 'rtiSW',
                 [['z', 'x2', 'z2N'], ['z2N', 'x2']]],
        'SWAP': ['rtiSW', 'rtiSW', [['Id'], ['x2N']], 'rtiSW', 'rtiSW', [['x2N'], ['Id']],
                 'rtiSW', 'rtiSW', [['Id'], ['x2N']]]
    }


class OneQCliffordTranspiler(CliffordUtils):
    def __init__(self, z_gate=True):
        if z_gate:
            self.onequbit_rule = self.cliffordGroupZ
        else:
            self.onequbit_rule = self.cliffordGroup
        self.one_qubit_clifford()

    def one_qubit_clifford(self):
        index = 1
        self.OneQubitClif = {}
        for i, iCli in enumerate(self.onequbit_rule.items()):
            self.OneQubitClif[index] = iCli[0]
            index += 1
        return self.OneQubitClif

    def cli2phys(self, cli_idx: int):
        # convert clifford gate to physical gate in the system
        if cli_idx <= 24:
            cligate = self.OneQubitClif[cli_idx]
            return self.onequbit_rule[cligate]
        else:
            ValueError("Clifford gate index out of range. The index should be between 1 to 24")

    def circuit_transpile(self, seq):
        phys_gate_list = []
        for iCli in seq:
            cli_phys = self.cli2phys(iCli)
            for jgate in cli_phys:
                phys_gate_list.append(jgate)
        return phys_gate_list


class TwoQCliffordTranspiler(CliffordUtils):
    def __init__(self, z_gate=True, rtiSW=False):
        if z_gate:
            self.onequbit_rule = self.cliffordGroupZ
        else:
            self.onequbit_rule = self.cliffordGroup
        if rtiSW:
            self.twoqubit_rule = self.twoQubitGroupRt
        else:
            self.twoqubit_rule = self.twoQubitGroup
        self.twoqubit_rule.update(self.onequbit_rule)
        self.twoqubit_rule.update(self.sGroup)
        self.two_qubit_clifford()

    def two_qubit_clifford(self):
        """
        Generate a dict that transpile all two qubit clifford gate into native gates in experiment.
        Two qubit clifford can be categorized into four groups:
        (1). combinations of single qubit clifford gate on two qubis (24*24=576),
        (2). single qubit clifford gate + CNOT + s gate (24*24*3*3=5184),
        (3). single qubit clifford gate + iSWAP + s gate (24*24*3*3=5184),
        (4). single qubit clifford gate + SWAP (24*24=576),
        :return: dict of two qubit clifford gate in four groups
        """
        index = 1
        self.TwoQubitClif = {}
        Group1 = {}
        for i, iCli in enumerate(self.onequbit_rule.items()):
            for j, jCli in enumerate(self.onequbit_rule.items()):
                Group1[f'G1_{i}_{j}'] = [[[iCli[0]], [jCli[0]]]]
                self.TwoQubitClif[index] = Group1[f'G1_{i}_{j}']
                index += 1

        Group2 = {}
        for i, iCli in enumerate(self.onequbit_rule.items()):
            for j, jCli in enumerate(self.onequbit_rule.items()):
                for k, kS in enumerate(self.sGroup.items()):
                    for l, lS in enumerate(self.sGroup.items()):
                        Group2[f'G2_{i}_{j}_{k}_{l}'] = [[[iCli[0]], [jCli[0]]], 'CNOT', [[kS[0]], [lS[0]]]]
                        self.TwoQubitClif[index] = Group2[f'G2_{i}_{j}_{k}_{l}']
                        index += 1

        Group3 = {}
        for i, iCli in enumerate(self.onequbit_rule.items()):
            for j, jCli in enumerate(self.onequbit_rule.items()):
                for k, kS in enumerate(self.sGroup.items()):
                    for l, lS in enumerate(self.sGroup.items()):
                        Group3[f'G3_{i}_{j}_{k}_{l}'] = [[[iCli[0]], [jCli[0]]], 'iSWAP', [[kS[0]], [lS[0]]]]
                        self.TwoQubitClif[index] = Group3[f'G3_{i}_{j}_{k}_{l}']
                        index += 1

        Group4 = {}
        for i, iCli in enumerate(self.onequbit_rule.items()):
            for j, jCli in enumerate(self.onequbit_rule.items()):
                Group4[f'G4_{i}_{j}'] = [[[iCli[0]], [jCli[0]]], 'SWAP']
                self.TwoQubitClif[index] = Group4[f'G4_{i}_{j}']
                index += 1

        return {'G1': Group1, 'G2': Group2, 'G3': Group3, 'G4': Group4}

    def cli2phys(self, cli_idx):
        # convert clifford gate to physical gate in the system

        def to_phys(gate, gate_list=None):
            if gate_list is None:
                gate_list = []
            gate_list_in = []
            for igate in gate:
                if isinstance(igate, str):
                    for jgate in self.twoqubit_rule[igate]:
                        gate_list_in.append(jgate)
                elif isinstance(igate, list):
                    to_phys(igate, gate_list_in)
            gate_list.append(gate_list_in)
            return gate_list

        if cli_idx <= 11520:
            cligate = self.TwoQubitClif[cli_idx]
            return to_phys(cligate)[0]
        else:
            ValueError("Clifford gate index out of range. The index should be between 1 to 11520")

    def circuit_transpile(self, seq):
        phys_gate_list = []
        for iCli in seq:
            cli_phys = self.cli2phys(iCli)
            for jgate in cli_phys:
                phys_gate_list.append(jgate)
        return phys_gate_list


class RandomizedBechmark1Q(OneQCliffordTranspiler):
    def __init__(self, cfg, seed: int=None, z_gate=True):
        self.cfg = cfg  # qubit
        self.z_gate = z_gate
        self.NUM_CLI = 24
        self.seed = seed
        random.seed(self.seed)
        super().__init__(z_gate=self.z_gate)

    def reset_random(self):
        random.seed(self.seed)

    def generate_rb_seq(self, length: int):
        seq = [random.randint(1, self.NUM_CLI) for i in range(length)]
        seq.append(self.recover_gate(seq))
        return seq

    def generate_irb_seq(self, length: int, gate_i: int | str):
        clif_idx = {'Id': 1, 'x': 2, 'y': 5, 'x2': 3, 'y2': 6}
        if type(gate_i) is str:
            if gate_i in clif_idx.keys():
                gate_i = clif_idx[gate_i]
            else:
                raise KeyError(f"{gate_i} is not supported. Choose the interleaved gate"
                               f" from [Id, x, y ,x2, y2] or input an index of Clifford gate")
        seq = []
        for i in range(length):
            seq.append(random.randint(1, self.NUM_CLI))
            seq.append(gate_i)
        seq.append(self.recover_gate(seq))
        # print("seq", seq)
        return seq

    def generate_rb_physeq(self, length: int):
        seq = self.generate_rb_seq(length)
        # print(self.circuit_transpile(seq))
        return self.circuit_transpile(seq)

    def generate_irb_physeq(self, length: int, gate_i: int | str):
        seq = self.generate_irb_seq(length, gate_i)
        return self.circuit_transpile(seq)

    def clifford_matrix(self, idx: int):
        gate_seq = self.onequbit_rule[self.OneQubitClif[idx]]
        U = qt.identity(2)
        for gate in gate_seq:
            U = qubitGateBasis[gate] * U
            # U *= qubitGateBasis[gate]
        return U

    # def recover_gate(self, seq):
    #     #make sure the whole unitary is Id. somehow it doesn't work for z gate
    #     U_seq = qt.identity(2)
    #     for gate in seq:
    #         u_gate = self.clifford_matrix(gate)
    #         U_seq *= u_gate
    #     for i in range(self.NUM_CLI):
    #         U_rec = self.clifford_matrix(i + 1) * U_seq
    #         # if U_rec[0, 0] != 0:
    #         if U_rec[0, 0] != 0 and U_rec / U_rec[0, 0] == qt.identity(2):
    #             return i + 1
    #     raise RuntimeError(f"cannot find recover gate for Clifford sequence {seq},{U_seq}")

    def recover_gate(self, seq):
        # only make sure the final state is |g><g|
        U_seq = qt.basis(2, 0) * qt.basis(2, 0).dag()
        for gate in seq:
            u_gate = self.clifford_matrix(gate)
            U_seq = u_gate * U_seq * u_gate.dag()
        for i in range(self.NUM_CLI):
            U_rec = self.clifford_matrix(i + 1) * U_seq * self.clifford_matrix(i + 1).dag()
            if U_rec == qt.basis(2, 0) * qt.basis(2, 0).dag():
                return i + 1
        raise RuntimeError(f"cannot find recover gate for Clifford sequence {seq},{U_seq}")


class RandomizedBechmark2Q(TwoQCliffordTranspiler):
    def __init__(self, cfg, seed: int=None):
        self.cfg = cfg  # qubit
        self.z_gate = True
        self.rtiSW = False
        self.NUM_CLI = 11520
        self.seed = seed
        random.seed(self.seed)
        super().__init__(z_gate=self.z_gate)

    def reset_random(self):
        random.seed(self.seed)

    def generate_rb_seq(self, length: int):
        seq = [random.randint(1, self.NUM_CLI) for i in range(length)]
        seq.append(self.recover_gate(seq))
        return seq

    def generate_irb_seq(self, length: int, gate_i: int | str):
        clif_idx = {'CNOT': 577, 'iSWAP': 5761, 'SWAP': 10945}
        if type(gate_i) is str:
            if gate_i in clif_idx.keys():
                gate_i = clif_idx[gate_i]
            else:
                raise KeyError(f"{gate_i} is not supported. Choose the interleaved gate"
                               f" from [CNOT, iSWAP, SWAP] or input an index of Clifford gate")
        seq = []
        for i in range(length):
            seq.append(random.randint(1, self.NUM_CLI))
            seq.append(gate_i)
        seq.append(self.recover_gate(seq))
        return seq

    def generate_rb_physeq(self, length: int):
        seq = self.generate_rb_seq(length)
        return self.circuit_transpile(seq)

    def generate_irb_physeq(self, length: int, gate_i: int | str):
        seq = self.generate_irb_seq(length, gate_i)
        return self.circuit_transpile(seq)

    def clifford_matrix(self, idx: int):
        def q1_matrix(glist: list):
            u = qt.identity(2)
            for g in glist:
                u *= qubitGateBasis[g]
            return u

        gate_seq = self.cli2phys(idx)
        U = qt.tensor(qt.identity(2), qt.identity(2))

        for gate in gate_seq:
            if type(gate) is str:
                U *= qubitGateBasis[gate]
            else:
                for i in range(2):
                    u_list = [qt.identity(2), qt.identity(2)]
                    u_list[i] = q1_matrix(gate[i])
                    U *= qt.tensor(u_list)
        return U

    def recover_gate(self, seq):
        U0 = qt.tensor(qt.identity(2), qt.identity(2))
        U_seq = U0
        for gate in seq:
            u_gate = self.clifford_matrix(gate)
            U_seq *= u_gate
        for i in range(self.NUM_CLI):
            U_rec = self.clifford_matrix(i + 1) * U_seq
            # if U_rec[0, 0] != 0:
            if U_rec[0, 0] != 0 and U_rec / U_rec[0, 0] == U0:
                return i + 1
        raise RuntimeError(f"cannot find recover gate for Clifford sequence {seq},{U_seq}")


class CliffordSeqProgram(QubitMsmtMixin, QCAveragerProgram, RandomizedBechmark1Q):
    def __init__(self, soccfg, cfg, qc_cfg):
        self.seed = cfg.get('seed', 0)
        RandomizedBechmark1Q.__init__(self, cfg, self.seed, z_gate=cfg.get('z_gate', False))
        QCAveragerProgram.__init__(self, soccfg, cfg, qc_cfg)

    def initialize(self):
        cfg = self.cfg
        self.gate_list = self.get_gate_seq(cfg["cli_list"], suffix="SH_q1", recover=cfg["recover"])

        self.res_ch = self.cfg["gen_chs"]["res_drive"]["ch"]
        self.qubit_ch = self.cfg["gen_chs"]["subh_drive"]["ch"]

        # set readout pulse registers
        self.set_pulse_params_auto_gen_type("res_drive", **cfg["res_pulse_config"])
        # add qubit pulse to q_drive channel
        self.add_waveform_from_cfg("q_drive", "q_gauss")
        self.add_waveform_from_cfg(cfg["q_gen"], "subh_pi")
        self.add_waveform_from_cfg(cfg["q_gen"], "subh_pi2")

        self.sync_all(self.us2cycles(1))  # give processor some time to configure pulses

    def get_gate_seq(self, seq, suffix: str=None, recover=True):
        """
        generate a gate sequence repeating a clifford gate N times, plus the recover gate
        :param length: length of the sequence
        :param gate: the repeated clifford gate
        :param suffix: suffix added to gate name
        :param recover: add a recover gate to flip qubit back to ground state
        :return: the gate sequence in pulse names available in pulse_config
        """
        if recover:
            seq.append(self.recover_gate(seq))
        self.cli_list = seq
        gate_seq = self.circuit_transpile(seq)
        if suffix:
            for i in range(len(gate_seq)):
                gate_seq[i] = gate_seq[i] + f"_{suffix}"
        return gate_seq

    def body(self):
        cfg = self.cfg
        self.reset_phaseOffset_dict()
        sel_msmt = cfg.get("sel_msmt", False)
        if sel_msmt:
            self.add_gate_by_name(f"x2_{cfg['qubit']}")
            self.sync_all(20)
            self.measure(pulse_ch=self.res_ch,
                         adcs=self.ro_chs,
                         pins=[0],
                         adc_trig_offset=self.cfg["adc_trig_offset"],
                         wait=True,
                         syncdelay=self.us2cycles(self.cfg["relax_delay"]))
        # for gate in self.gate_list:
        #     temp = self.add_gate_by_name(f"{gate}")
        self.add_gate_concatenate(self.gate_list, f"RB_{len(self.cli_list)}")

        self.sync_all(20)
        self.measure(pulse_ch=self.res_ch,
                     adcs=self.ro_chs,
                     pins=[0],
                     adc_trig_offset=self.cfg["adc_trig_offset"],
                     wait=True,
                     syncdelay=self.us2cycles(self.cfg["relax_delay"]))


class CliffordCircuitProgram(QubitMsmtMixin, CircuitProgram, RandomizedBechmark1Q):
    """
    CircuitProgram-based replacement for CliffordSeqProgram: runs a fixed cli_list
    circuit (+ optional recover gate, + optional sel_msmt selection measurement).
    Same semantics as CliffordSeqProgram, but:
      - the gate suffix is derived from cfg['qubit'] (CliffordSeqProgram hardcodes
        "SH_q1", which silently breaks for any qubit other than q1);
      - phase correction is resolved per-gate "type" via GateLibrary instead of
        being hardcoded into the base class (see DEFAULT_PHASE_CORRECTIONS below,
        and qc_cfg['pulse_config'] entries must carry a matching "type" field:
        "subharmonic_3" for *_SH_{qubit} gates, "physical" for direct-drive gates
        like the x2_{qubit} selection pulse -- leaving both untyped would collide
        them onto the same phase correction).

    Caveats inherited from Hatlab_RFSOC.core.circuit_program (see its module
    docstring), not re-verified here:
      - PhaseCorrection.correction_deg uses absolute channel time, which already
        reflects prior pulses' durations, while frame_update_deg also adds a
        duration-based term to the running frame afterward -- whether this
        double-counts hasn't been numerically checked.
      - add_sequence_concatenated advances its internal time accumulator using
        each gate's configured waveform length (us), not the actual registered
        envelope's cycle-rounded length, which the old add_gate_concatenate used --
        a small precision difference that could matter for long Clifford sequences.
    """

    DEFAULT_PHASE_CORRECTIONS: Dict[str, PhaseCorrection] = {
        GateLibrary.PHYSICAL: NullPhaseCorrection(),           # direct-drive gates (e.g. x2_{qubit} selection pulse)
        "subharmonic_3": SubharmonicPhaseCorrection(order=3),  # *_SH_{qubit} circuit gates
    }

    def __init__(self, soccfg, cfg, qc_cfg, phase_corrections: Dict[str, PhaseCorrection] = None):
        self.seed = cfg.get('seed', 0)
        RandomizedBechmark1Q.__init__(self, cfg, self.seed, z_gate=cfg.get('z_gate', False))
        merged_corrections = {**self.DEFAULT_PHASE_CORRECTIONS, **(phase_corrections or {})}
        CircuitProgram.__init__(self, soccfg, cfg, qc_cfg, merged_corrections)

    def initialize(self):
        cfg = self.cfg
        suffix = f"SH_{cfg['qubit']}"
        self.gate_list = self.get_gate_seq(cfg["cli_list"], suffix=suffix, recover=cfg["recover"])

        self.res_ch = self.cfg["gen_chs"]["res_drive"]["ch"]
        self.qubit_ch = self.cfg["gen_chs"]["subh_drive"]["ch"]

        # set readout pulse registers
        self.set_pulse_params_auto_gen_type("res_drive", **cfg["res_pulse_config"])
        # add qubit pulse to q_drive channel
        self.add_waveform_from_cfg("q_drive", "q_gauss")
        self.add_waveform_from_cfg(cfg["q_gen"], "subh_pi")
        self.add_waveform_from_cfg(cfg["q_gen"], "subh_pi2")

        self.sync_all(self.us2cycles(1))  # give processor some time to configure pulses

    def get_gate_seq(self, seq, suffix: str = None, recover=True):
        """
        generate a gate sequence repeating a clifford gate N times, plus the recover gate
        :param length: length of the sequence
        :param gate: the repeated clifford gate
        :param suffix: suffix added to gate name
        :param recover: add a recover gate to flip qubit back to ground state
        :return: the gate sequence in pulse names available in pulse_config
        """
        if recover:
            seq.append(self.recover_gate(seq))
        self.cli_list = seq
        gate_seq = self.circuit_transpile(seq)
        if suffix:
            gate_seq = [f"{g}_{suffix}" for g in gate_seq]
        return gate_seq

    def body(self):
        cfg = self.cfg
        self.frame.reset()
        sel_msmt = cfg.get("sel_msmt", False)
        if sel_msmt:
            self.add_gate(f"x2_{cfg['qubit']}")
            self.sync_all(20)
            self.measure(pulse_ch=self.res_ch,
                         adcs=self.ro_chs,
                         pins=[0],
                         adc_trig_offset=self.cfg["adc_trig_offset"],
                         wait=True,
                         syncdelay=self.us2cycles(self.cfg["relax_delay"]))

        # None by default -> identical (non-modulated) pulse-building path as CliffordSeqProgram;
        # pass cfg["circuit_modulations"] = ["chirp"] (+ cfg["modulations"]["chirp"] definition) to
        # opt into e.g. chirp modulation without needing a separate subclass.
        modulations = cfg.get("circuit_modulations")
        self.add_sequence_concatenated(self.gate_list, f"RB_{len(self.cli_list)}", modulations=modulations)

        self.sync_all(20)
        self.measure(pulse_ch=self.res_ch,
                     adcs=self.ro_chs,
                     pins=[0],
                     adc_trig_offset=self.cfg["adc_trig_offset"],
                     wait=True,
                     syncdelay=self.us2cycles(self.cfg["relax_delay"]))



if __name__ == "__main__":
    cfg = {}
    ct = OneQCliffordTranspiler()
    rb1 = RandomizedBechmark1Q(cfg, 0, z_gate=False)
    rb1z = RandomizedBechmark1Q(cfg, 0)

