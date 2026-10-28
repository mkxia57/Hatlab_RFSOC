"""
Circuit-level qick program, restructured as a set of separately testable pieces
instead of one class that does gate lookup, phase bookkeeping, and waveform
building all at once (that's `QCAveragerProgram` in averager_program.py, which
stays put for backward compatibility).

Layering
--------
GateLibrary       resolves gate names to gate specs. Pure lookup, no hardware access.
PhaseCorrection   strategy interface for drive/qubit frequency-mismatch phase math
                  (e.g. subharmonic driving). Swap the implementation instead of
                  editing formulas inline wherever a pulse gets built.
PhaseFrameTracker plain per-qubit running virtual-frame phase accumulator - no
                  opinion about gate types or which PhaseCorrection applies.
CircuitProgram    thin orchestrator: resolve gate -> pick this gate's PhaseCorrection
                  by its "type" (CircuitProgram.phase_corrections) -> phase-correct
                  -> ensure waveform exists -> play. Waveform construction (single or
                  concatenated, with optional modulations such as chirp/drag) is
                  delegated entirely to Hatlab_RFSOC.waveform.waveform - not
                  reimplemented here.

Caveat
------
The phase-correction formulas in `SubharmonicPhaseCorrection` and the gain
normalization in `CircuitProgram.add_sequence_concatenated` are ported from
`QCAveragerProgram._pulse_phaseOffset` / `update_pulse_phase` /
`add_gate_concatenate`, not re-derived. `add_sequence_concatenated` uses each
physical gate's configured `length` (in us) to advance the running phase, whereas the original
measured the actual registered envelope's cycle-rounded length - a small
precision difference. Validate against the old class on a known sequence before
relying on this for real measurements.

Known open question (not fixed here, see PhaseFrameTracker.accumulate_offsets):
`PhaseCorrection.correction_deg` is recomputed fresh each gate from the
*absolute* channel time, which already reflects every prior pulse's duration
on that channel. `frame_update_deg` then *also* adds a pulse-duration-based
term to the running per-qubit offset after every physical gate. Whether this
double-counts the pulse-duration contribution hasn't been verified
numerically - flagged for a follow-up trace, not resolved in this pass.
"""
from copy import deepcopy
from typing import Dict, Iterable, List, Optional, Union
import warnings

from Hatlab_RFSOC.core.averager_program import NDAveragerProgram
from Hatlab_RFSOC.waveform.waveform import add_waveform_concatenate

GateSpec = Dict  # a resolved, mutable-safe copy of one qc_cfg["pulse_config"][name] entry


# --------------------------------------------------------------------------------------
# Phase correction strategy
# --------------------------------------------------------------------------------------

class PhaseCorrection:
    """Strategy interface for the phase correction owed to a drive/qubit frequency
    mismatch. Isolates platform-specific phase math so it lives in one place instead
    of being copy-pasted across every gate-application method."""

    def correction_deg(self, gate: GateSpec, qubit_freq_ge: float, t0_us: float) -> float:
        """Phase (deg) to add to `gate['phase']`, given the pulse starts at `t0_us`
        (us, generator-channel clock) and the qubit's ge frequency is `qubit_freq_ge`
        (MHz)."""
        raise NotImplementedError

    def frame_update_deg(self, gate: GateSpec, qubit_freq_ge: float, pulse_len_us: float) -> float:
        """How much the qubit's running virtual frame should shift after this gate
        plays, e.g. because the drive is not co-rotating with the qubit."""
        return 0.0


class NullPhaseCorrection(PhaseCorrection):
    """No-op. Use when the drive frequency always matches the qubit frequency."""

    def correction_deg(self, gate, qubit_freq_ge, t0_us):
        return 0.0


class SubharmonicPhaseCorrection(PhaseCorrection):
    """Phase correction for an `order`-th subharmonic drive (drive freq * order !=
    qubit freq). `order=3` reproduces the behavior hardcoded in the old
    `QCAveragerProgram` (see the "todo: only works for 4-wave subharmonic, should
    make it general" comments it carried in three separate places)."""

    def __init__(self, order: int = 3):
        self.order = order
        self.wrap_deg = 360 / order

    def _freq_diff(self, gate, qubit_freq_ge):
        return qubit_freq_ge - self.order * gate["freq"]

    def correction_deg(self, gate, qubit_freq_ge, t0_us):
        freq_diff = self._freq_diff(gate, qubit_freq_ge)
        return 360 * freq_diff / self.order * t0_us

    def frame_update_deg(self, gate, qubit_freq_ge, pulse_len_us):
        freq_diff = self._freq_diff(gate, qubit_freq_ge)
        phi1 = freq_diff / self.order * pulse_len_us * 360
        return -phi1 % self.wrap_deg


class PhaseFrameTracker:
    """Plain per-qubit running virtual drive-frame phase accumulator. Holds no
    opinion about gate types, PhaseCorrection selection, or virtual-Z sign
    conventions - all of that is domain logic owned by CircuitProgram (which
    knows about gate types and which PhaseCorrection applies to a given gate),
    assembled into a per-qubit phase_offsets dict and handed to
    accumulate_offsets. Keeping this class this dumb is what let virtual-Z's
    "-phase" convention and the calibrated/frequency-mismatch corrections move
    to CircuitProgram without this class needing to change."""

    def __init__(self, qubits: Iterable[str]):
        self._offset = {q: 0.0 for q in qubits}

    def get_offset(self, qubit: str) -> float:
        return self._offset[qubit]

    def accumulate_offsets(self, phase_offsets: Dict[str, float]):
        """Add a per-qubit phase offset (deg) to each qubit's running frame."""
        for qubit, offset in phase_offsets.items():
            self._offset[qubit] += offset

    def reset(self):
        self._offset = {q: 0.0 for q in self._offset}


# --------------------------------------------------------------------------------------
# Gate lookup
# --------------------------------------------------------------------------------------

class GateLibrary:
    """Resolves gate names to gate specs, using an explicit "type" field so
    dispatch (physical pulse vs. virtual Z) doesn't depend on name conventions.
    Existing configs without a "type" field fall back to the legacy convention
    (name starts with "z" => virtual Z), with a warning, so old yaml configs keep
    working while you migrate them to the explicit field.

    "type" is not just a physical/virtual-Z switch: any value other than
    VIRTUAL_Z means "this gate builds a pulse, and this exact string is the
    CircuitProgram.phase_corrections registry key for it" - e.g. "physical"
    (the legacy/untyped fallback), "subharmonic_3", "iswap_2", etc. Different
    gates for the same qubit can need different PhaseCorrection formulas (a
    direct-drive gate needs none, a subharmonic-drive gate does), so the type
    string doubles as that lookup key rather than adding a second field."""

    VIRTUAL_Z = "virtual_z"
    PHYSICAL = "physical"  # legacy/untyped fallback value; also the default PhaseCorrection key

    def __init__(self, qc_cfg: dict):
        self.qc_cfg = qc_cfg

    def resolve(self, gate_name: str) -> GateSpec:
        """Return a fresh copy of the gate's config, safe for the caller to mutate."""
        if gate_name not in self.qc_cfg["pulse_config"]:
            raise KeyError(f"gate '{gate_name}' not found in qc_cfg['pulse_config']")
        return deepcopy(self.qc_cfg["pulse_config"][gate_name])

    def gate_type(self, gate_name: str) -> str:
        """Return this gate's "type" string. VIRTUAL_Z means frame-only, no
        pulse; anything else means "builds a pulse, using this string as the
        CircuitProgram.phase_corrections registry key" (e.g. "physical",
        "subharmonic_3", "iswap_2", ...)."""
        gate = self.qc_cfg["pulse_config"][gate_name]
        if "type" in gate:
            return gate["type"]
        warnings.warn(
            f"gate '{gate_name}' has no explicit 'type' field; falling back to "
            f"name-based inference. Add an explicit 'type' (e.g. 'virtual_z', "
            f"'physical', 'subharmonic_3', ...) to qc_cfg['pulse_config']['{gate_name}'] "
            f"to silence this and select the right PhaseCorrection.",
            DeprecationWarning,
        )
        return self.VIRTUAL_Z if gate_name[0] == "z" else self.PHYSICAL

    def qubit_of(self, gate_name: str) -> str:
        return self.qc_cfg["pulse_config"][gate_name]["qubit"]

    def phase_offsets(self, gate: GateSpec) -> Dict[str, float]:
        """Per-qubit calibrated phase offset (deg) owed for playing this gate,
        e.g. {"q2": 110.679}. A gate's physical pulse parameters (gen_ch, freq,
        gain, waveform, style) describe one hardware channel and stay scalar;
        this is the one place a gate's consequences can span multiple qubits
        (e.g. a future two-qubit gate touching two qubits' frames at once).

        Falls back to the legacy flat `qubit`/`phaseOffset` fields when no
        explicit `phase_offsets` dict is present on the gate, so existing yaml
        configs (which calibrate a single float per gate) need no changes."""
        if "phase_offsets" in gate:
            return dict(gate["phase_offsets"])
        return {gate["qubit"]: gate.get("phaseOffset", 0.0)}


# --------------------------------------------------------------------------------------
# Circuit program
# --------------------------------------------------------------------------------------

class CircuitProgram(NDAveragerProgram):
    """Circuit-level qick program: sequences named gates from
    `qc_cfg['pulse_config']` onto generator channels, tracking each qubit's virtual
    drive-frame phase. Restructured replacement for `QCAveragerProgram`.

    Measurement helpers (state-prep msmt, tomography) are intentionally not
    reimplemented here - mix in `QubitMsmtMixin` from averager_program.py alongside
    this class if you need them, rather than duplicating a second tomo()/prepare_msmt
    implementation on the circuit class itself.
    """

    PULSE_PARAMS = ("gen_ch", "style", "freq", "phase", "gain", "phrst", "stdysel",
                    "mode", "outsel", "length", "waveform", "mask")

    def __init__(self, soccfg, cfg, qc_cfg: dict, phase_corrections: Dict[str, PhaseCorrection]):
        self.qc_cfg = qc_cfg
        self.gates = GateLibrary(qc_cfg)
        self.frame = PhaseFrameTracker(qc_cfg["qubit_config"].keys())
        self.phase_corrections = phase_corrections
        self._last_pulse_params: Dict[int, GateSpec] = {}
        super().__init__(soccfg, cfg)

    def _ch(self, gen_ch: Union[str, int]) -> int:
        return self.cfg["gen_chs"][gen_ch]["ch"] if isinstance(gen_ch, str) else gen_ch

    def _strip_aux_params(self, gate: GateSpec) -> GateSpec:
        return {k: v for k, v in gate.items() if k in self.PULSE_PARAMS}

    def _correction_for(self, gate_type: str) -> PhaseCorrection:
        """Look up the PhaseCorrection registered for a (non-virtual-Z) gate type."""
        try:
            return self.phase_corrections[gate_type]
        except KeyError:
            raise KeyError(
                f"no PhaseCorrection registered for gate type '{gate_type}' "
                f"(available: {list(self.phase_corrections)})"
            ) from None

    def _apply_virtual_z(self, qubit: str, phase_deg: float):
        """Virtual-Z redefines the frame rather than physically rotating, so it
        reduces the qubit's running offset by the gate's phase."""
        self.frame.accumulate_offsets({qubit: -phase_deg})

    def _set_pulse_params_if_changed(self, ch: int, params: GateSpec):
        """Skip the (redundant) register-load instructions set_pulse_params
        would otherwise emit when this channel's params are identical to the
        last call on it - pulse() replays the existing registers correctly
        either way, so nothing is lost by not re-issuing them."""
        if self._last_pulse_params.get(ch) != params:
            self.set_pulse_params(**params)
            self._last_pulse_params[ch] = params

    def _resolve_modulations(self, gate: GateSpec, modulations: List[str]) -> List[str]:
        """Substitute a gate-level override into any requested modulation whose
        cfg['modulations'] entry needs one, e.g. a per-gate `chirp` field (real
        configs calibrate a distinct detune per gate, like x_SH_q4's -39.367 vs
        x2_SH_q4's -39.517) overriding a shared ChirpModulation's fixed `maxf`.
        Registers a uniquely-named variant in cfg['modulations'] the first time
        a given (modulation name, override value) pair is seen and reuses it
        after. Modulations with no matching gate override pass through as-is."""
        if "chirp" not in gate:
            return modulations
        all_mods = self.cfg.setdefault("modulations", {})
        resolved = []
        for name in modulations:
            mod_cfg = all_mods.get(name, {})
            if mod_cfg.get("type", "").lower() != "chirpmodulation":
                resolved.append(name)
                continue
            variant_name = f"{name}__chirp_{gate['chirp']}"
            if variant_name not in all_mods:
                all_mods[variant_name] = {**mod_cfg, "maxf": gate["chirp"]}
            resolved.append(variant_name)
        return resolved

    def add_gate(self, gate_name: str, phase_offset: float = 0):
        """Play a single named gate. Virtual-Z gates only update the running frame."""
        gtype = self.gates.gate_type(gate_name)
        if gtype == GateLibrary.VIRTUAL_Z:
            gate = self.gates.resolve(gate_name)
            self._apply_virtual_z(gate["qubit"], gate["phase"])
            return

        gate = self.gates.resolve(gate_name)
        ch = self._ch(gate["gen_ch"])
        qubit = gate["qubit"]
        qubit_freq = self.qc_cfg["qubit_config"][qubit]["freq_ge"]
        t0_us = self.cycles2us(self._gen_ts[ch])
        correction = self._correction_for(gtype)

        gate["phase"] = (gate.get("phase", 0)
                          + self.frame.get_offset(qubit)
                          + correction.correction_deg(gate, qubit_freq, t0_us)
                          + phase_offset)

        if gate["style"] in ("arb", "flat_top"):
            if gate["waveform"] not in self.envelopes[ch]["envs"]:
                self.add_waveform_from_cfg(gate["gen_ch"], gate["waveform"])
        elif gate["style"] != "const":
            raise ValueError(f"unsupported pulse style '{gate['style']}' for gate '{gate_name}'")

        pulse_len_us = self.cfg["waveforms"][gate["waveform"]]["length"] if "waveform" in gate else 0.0
        offsets = self.gates.phase_offsets(gate)
        offsets[qubit] = offsets.get(qubit, 0.0) + correction.frame_update_deg(gate, qubit_freq, pulse_len_us)
        self.frame.accumulate_offsets(offsets)

        self._set_pulse_params_if_changed(ch, self._strip_aux_params(gate))
        self.pulse(ch)

    def add_sequence(self, gate_seq: List[str], phase_offset: float = 0):
        """Play every gate in `gate_seq`, in order, one at a time via add_gate -
        i.e. as separate pulses, not one concatenated waveform. Use this for a
        short/one-off sequence, for debugging, or when the real hardware gaps
        between pulses should stay separate pulses rather than being folded
        into one contiguous envelope. Use add_sequence_concatenated instead
        for a sequence that gets replayed many times (e.g. a fixed RB Clifford
        sequence run over many averages), where building the waveform once
        pays off.
        """
        for gate_name in gate_seq:
            self.add_gate(gate_name, phase_offset=phase_offset)

    def add_sequence_concatenated(self, gate_seq: List[str], seq_name: str, phase_offset: float = 0,
                      modulations: Optional[List[str]] = None):
        """Build (once) and play a single concatenated waveform for every physical
        gate in `gate_seq`, in order; virtual-Z gates in the sequence only
        contribute frame bookkeeping. `modulations` (names looked up in
        `cfg['modulations']`) are applied to every physical gate's envelope -
        pass e.g. `modulations=["chirp"]` for what used to be a separate
        "chirp concatenate" method. A gate with its own `chirp` field
        overrides a shared ChirpModulation's detune for that gate only, via
        `_resolve_modulations` - see real configs where e.g. x_SH_q4 and
        x2_SH_q4 calibrate different chirp values under the same modulation name.
        """
        gate0 = self.gates.resolve(gate_seq[0])
        ch = self._ch(gate0["gen_ch"])
        qubit = gate0["qubit"]
        qubit_freq = self.qc_cfg["qubit_config"][qubit]["freq_ge"]

        physical_gates = [g for g in gate_seq if self.gates.gate_type(g) != GateLibrary.VIRTUAL_Z]
        gains = [self.gates.resolve(g).get("gain", 0) for g in physical_gates]
        gmax = max(gains) if gains else 0

        if seq_name not in self.envelopes[ch]["envs"]:
            wf_list = []
            # Running channel time for this sequence's gates, mirroring the
            # legacy add_gate_concatenate's `t0 += pulse_cycle` accumulator:
            # each gate's correction_deg must use the time *it* will start at
            # once the concatenated waveform plays, not the sequence's start
            # time reused for every gate.
            t_us = self.cycles2us(self._gen_ts[ch])
            for gate_name in gate_seq:
                gtype = self.gates.gate_type(gate_name)
                if gtype == GateLibrary.VIRTUAL_Z:
                    gate = self.gates.resolve(gate_name)
                    self._apply_virtual_z(qubit, gate["phase"])
                    continue

                gate = self.gates.resolve(gate_name)
                if gate.get("gain", 0) == 0:
                    continue  # identity / zero-amplitude placeholder gate: no waveform segment

                correction = self._correction_for(gtype)
                gate["phase"] = (gate.get("phase", 0) + self.frame.get_offset(qubit)
                                  + correction.correction_deg(gate, qubit_freq, t_us)
                                  + phase_offset)

                wf_cfg = deepcopy(self.cfg["waveforms"][gate["waveform"]])
                wf_cfg["phase"] = gate["phase"]
                if gmax:
                    base_gencfg = self.soccfg["gens"][ch]
                    base_maxv = base_gencfg["maxv"] * base_gencfg["maxv_scale"]
                    wf_cfg["maxv"] = base_maxv * gate["gain"] / gmax
                if modulations:
                    wf_cfg["modulations"] = self._resolve_modulations(gate, modulations)
                wf_list.append(wf_cfg)

                pulse_len_us = wf_cfg["length"]
                offsets = self.gates.phase_offsets(gate)
                offsets[qubit] = offsets.get(qubit, 0.0) + correction.frame_update_deg(gate, qubit_freq, pulse_len_us)
                self.frame.accumulate_offsets(offsets)
                t_us += pulse_len_us

            add_waveform_concatenate(self, ch, seq_name, wf_list)

        gate0["waveform"] = seq_name
        gate0["gain"] = gmax if gmax else gate0.get("gain", 0)
        self._set_pulse_params_if_changed(ch, self._strip_aux_params(gate0))
        self.pulse(ch)

    def add_x(self, qubit: str, **kwargs):
        self.add_gate(f"x_{qubit}", **kwargs)

    def add_x2(self, qubit: str, **kwargs):
        self.add_gate(f"x2_{qubit}", **kwargs)

    def add_y(self, qubit: str, **kwargs):
        self.add_gate(f"y_{qubit}", **kwargs)

    def add_y2(self, qubit: str, **kwargs):
        self.add_gate(f"y2_{qubit}", **kwargs)

    def add_iswap(self, q_drive_ch: str, pulse_cfg: dict = None):
        raise NotImplementedError("two-qubit iSWAP gate is not implemented yet in CircuitProgram")

    def add_rtiswap(self, q_drive_ch: str, pulse_cfg: dict = None):
        raise NotImplementedError("two-qubit resonant/tunable iSWAP gate is not implemented yet in CircuitProgram")