from typing import List, Dict, Union
import numpy as np

from plottr.data.datadict import DataDict, DataDictBase
from Hatlab_DataProcessing.data_saving import datadict_from_hdf5


class QickDataDict(DataDict):
    """
    Subclass of plottr.DataDict class for keeping data from "QickProgram"s
    """

    def __init__(self, ro_chs, inner_sweeps: DataDictBase = None, outer_sweeps: DataDictBase = None):
        """
        initialize the DataDict class with sweep axes.
        :param ro_chs:
        :param inner_sweeps: DataDict that contains the axes of inner sweeps. In the initialization, the DataDict can
            just provide the names of inner sweeps axes and their units, the axes values can be empty. If the inner
            sweep value does not change for each outer sweep, the values of each inner sweep axes can also be provided,
            in which case the order of items in the dict has to follow first->last : innermost_sweep-> outermost_sweep.
        :param outer_sweeps: ataDict that contains the axes of outer sweeps. Again, axes values can be empty.
        """
        if outer_sweeps is None:
            outer_sweeps = {}
        if inner_sweeps is None:
            inner_sweeps = {}

        self.ro_chs = ro_chs
        self.outer_sweeps = outer_sweeps
        self.inner_sweeps = inner_sweeps

        dd = {"msmts": {}}
        dd["__val_msmts__"] = None

        for k, v in outer_sweeps.items():
            dd[k] = {"unit": v.get("unit")}
            dd[f"__val_{k}__"] = v.get("values")

        dd["reps"] = {}
        dd["__val_reps__"] = None

        for k, v in inner_sweeps.items():
            dd[k] = {"unit": v.get("unit")}
            dd[f"__val_{k}__"] = v.get("values")

        dd["soft_reps"] = {}
        dd["__val_soft_reps__"] = None

        for ch in ro_chs:
            dd[f"avg_iq_{ch}"] = {
                "axes": ["soft_reps", *list(outer_sweeps.keys())[::-1], "reps", *list(inner_sweeps.keys())[::-1],
                         "msmts"],
                # in the order of outer to inner axes
                "unit": "a.u.",
                "__isdata__": True
            }
            dd[f"buf_iq_{ch}"] = {
                "axes": ["soft_reps", *list(outer_sweeps.keys())[::-1], "reps", *list(inner_sweeps.keys())[::-1],
                         "msmts"],
                # in the order of outer to inner axes
                "unit": "a.u.",
                "__isdata__": True
            }
        super().__init__(**dd)

    def add_data(self, avg_i, avg_q, buf_i=None, buf_q=None,
                 inner_sweeps: Union[DataDict, DataDictBase, Dict] = None, soft_rep=0, **outer_sweeps) -> None:
        """
        Function for adding data to DataDict after each qick tproc inner sweep.

        At this point, there no requirement on the order of soft_rep and outer sweeps, e.g. you can sweep over all
        outer sweeps in whatever order, then do soft repeat (repeat in python), or, you can do soft repeat of the qick
        inner sweeps first, then sweep the outer parameters. The data-axes mapping relation will always be correct.

        BUT, to make the data extraction methods in"DataFromQDDH5" work correctly, it is recommended to add data in
        the order of outer_sweeps dict first (first key->last key), then add soft repeats.

        :param avg_i: averaged I data returned from qick.RAveragerProgram.acquire()
            (or other QickPrograms that uses the same data shape: (ro_ch, msmts, expts))
        :param avg_q: averaged Q data returned from qick.RAveragerProgram.acquire()
            (or other QickPrograms that uses the same data shape: (ro_ch, msmts, expts)
        :param buf_i: all the I data points measured in qick run.
            shape: (ro_ch, tot_reps, msmts_per_rep), where the order of points in the last dimension follows:(m0_exp1, m1_exp1, m0_exp2...)
        :param buf_q: all the Q data points measured in qick run.
            shape: (ro_ch, tot_reps, msmts_per_rep), where the order of points in the last dimension follows:(m0_exp1, m1_exp1, m0_exp2...)
        :param inner_sweeps: Dict or DataDict that contains the keys and values of each qick inner sweep. The order has
            to be first->last : innermost_sweep-> outermost_sweep. When the inner sweep values change for each new outer
            sweep value, the inner sweep values can be re-specified when each time we add data, otherwise, the values
            provided in initialize will be used.
        :param soft_rep: soft repeat index
        :param outer_sweeps: kwargs for the new outer sweep values used in this data acquisition run.

        :return:
        """
        new_data = {}
        msmt_per_exp = avg_i.shape[-2]
        reps = 1 if buf_i is None else buf_i.shape[-2]
        if inner_sweeps is None:
            inner_sweeps = self.inner_sweeps
        flatten_inner = flattenSweepDict(inner_sweeps)  # assume inner sweeps have a square shape
        expts = len(list(flatten_inner.values())[0])  # total inner sweep points

        # add msmt index data
        new_data["msmts"] = np.tile(range(msmt_per_exp), expts * reps)
        self["__val_msmts__"] = np.arange(msmt_per_exp)

        # add iq data
        for i, ch in enumerate(self.ro_chs):
            new_data[f"avg_iq_{ch}"] = np.tile((avg_i[i] + 1j * avg_q[i]).transpose().flatten(), reps)
            if buf_i is not None:
                new_data[f"buf_iq_{ch}"] = (buf_i[i] + 1j * buf_q[i]).flatten()
            else:
                new_data[f"buf_iq_{ch}"] = np.zeros(msmt_per_exp * expts)

        # add qick repeat index data
        new_data["reps"] = np.repeat(np.arange(reps), msmt_per_exp * expts)
        self["__val_reps__"] = np.arange(reps)

        # add qick inner sweep data
        for k, v in flatten_inner.items():
            new_data[k] = np.tile(np.repeat(v, msmt_per_exp), reps)

        # add outer sweep data
        for k, v in outer_sweeps.items():
            new_data[k] = np.repeat([v], msmt_per_exp * expts * reps)

        # add soft repeat index data
        new_data["soft_reps"] = np.repeat([soft_rep], msmt_per_exp * expts * reps)
        self["__val_soft_reps__"] = np.arange(soft_rep + 1)

        super().add_data(**new_data)


def flattenSweepDict(sweeps: Union[DataDictBase, Dict]):
    """
    Flatten a square sweep dictionary to 1d arrays.

    :param sweeps: dictionary of sweep variable arrays
    :return:
    """
    try:
        py_dict = sweeps.to_dict()
    except AttributeError:
        py_dict = sweeps

    flatten_sweeps = {}
    sweep_vals = map(np.ndarray.flatten, np.meshgrid(*py_dict.values()))
    for k in sweeps.keys():
        flatten_sweeps[k] = next(sweep_vals)
    return flatten_sweeps


class DataFromQDDH5:
    def __init__(self, ddh5_path, merge_reps=True, progress=False, fast_load=True):
        """
        load data from a DDH5 file that was created from a QickDataDict object. Adds the loaded data to dictionaries
        that are easy to use (avg_iq, buf_iq, axes). To ensure the correct order of axis values, the original data must
        be created in the order of (outer->inner): (soft_rep, outer_sweeps, reps, inner_sweeps, msmts)

        :param ddh5_path: path to the ddh5 file.
        :param merge_reps: when True, the soft_reps and reps (qick inner reps) will be merged into one axes. For avg_iq,
            the data from different soft repeat cycles will be averaged.
        :param progress: when True, show a progress bar for data loading.
        :param fast_load: when True, load the experiment data (avg_iq, buf_iq) only, and axes values will be loaded from
            metadata.
        """
        self.datadict = datadict_from_hdf5(ddh5_path, progress=progress, data_only=fast_load)
        self.avg_iq = {}
        self.buf_iq = {}
        self.axes = {}
        self.ro_chs = []
        self.reps = self.datadict.meta_val("val_reps")[-1] + 1
        self.soft_reps = self.datadict.meta_val("val_soft_reps")[-1] + 1
        self.total_reps = self.reps * self.soft_reps
        self.axes_names = []
        self.datashape = []

        # reshape original data based on the size of each sweep axes (including reps and msmts)
        for k, v in self.datadict.items():
            if "avg_iq" in k:
                rch = k.replace("avg_iq_", "")
                self.avg_iq[rch] = self._reshape_original_data(v)
                self.ro_chs.append(rch)
            if "buf_iq" in k:
                rch = k.replace("buf_iq_", "")
                self.buf_iq[rch] = self._reshape_original_data(v)

        if merge_reps:
            self._merge_reps()
        else:
            rep_idx = self.axes_names.index("reps")
            for k, v in self.avg_iq.items():
                self.avg_iq[k] = np.moveaxis(v, rep_idx, 0)[0]

        print("buffer data shape: ", self.datashape)
        print("buffer data axes: ", self.axes_names)

    def _reshape_original_data(self, data):
        """
        reshape original data based on the size of each sweep axes (including reps and msmts), and get the values of
        each sweep axes.

        :param data:
        :return:
        """
        data_shape = []
        if self.axes_names == []:
            self.axes_names = data["axes"]
        for ax in data["axes"]:
            try:  # assume all the sweep axis' values have been saved in metadata
                # get values of each sweep axes from metadata
                ax_val = self.datadict.meta_val(f"val_{ax}")
                if ax not in self.axes:  # only need to add once
                    self.axes[ax] = {"unit": self.datadict[ax].get("unit"), "values": ax_val}
                data_shape.append(len(ax_val))
            except KeyError:
                pass

        data_r = np.array(data["values"]).reshape(*data_shape)
        self.datashape = list(data_r.shape)

        return data_r

    def _merge_reps(self):
        """
        merge the software repeats and the qick inner repeats into one axes.
        :return:
        """
        rep_idx = self.axes_names.index("reps")
        for k, v in self.avg_iq.items():
            v = np.moveaxis(v, rep_idx, 1)
            self.avg_iq[k] = np.average(v.reshape(-1, *v.shape[2:]), axis=0)
        for k, v in self.buf_iq.items():
            v = np.moveaxis(v, rep_idx, 1)
            self.buf_iq[k] = v.reshape(-1, *v.shape[2:])
        self.datashape = list(self.buf_iq[k].shape)

        self.axes_names.pop(rep_idx)
        self.axes_names[0] = "reps"

        _new_axes = {"reps": np.arange(self.total_reps)}
        for k in self.axes_names[1:]:
            _new_axes[k] = self.axes[k]
        self.axes = _new_axes



if __name__ == "__main__":
    from plottr.apps.autoplot import autoplot

    ro_chs = [0, 1]
    n_msmts = 2
    reps = 3

    # make fake data
    inner_sweeps = DataDictBase(length={"unit": "ns", "values": np.linspace(0, 100, 11)},
                                phase={"unit": "deg", "values": np.linspace(0, 90, 2)}
                                )

    x1_pts = inner_sweeps["length"]["values"]
    x2_pts = inner_sweeps["phase"]["values"]

    avgi = np.zeros((len(ro_chs), n_msmts, len(x1_pts) * len(x2_pts)))
    avgq = np.zeros((len(ro_chs), n_msmts, len(x1_pts) * len(x2_pts)))
    bufi = np.zeros((len(ro_chs), reps, n_msmts * len(x1_pts) * len(x2_pts)))
    bufq = np.zeros((len(ro_chs), reps, n_msmts * len(x1_pts) * len(x2_pts)))

    for i, ch in enumerate(ro_chs):
        for m in range(n_msmts):
            avgi[i, m] = (flattenSweepDict(inner_sweeps)["length"] + flattenSweepDict(inner_sweeps)["phase"]) * (
                        m + 1) + i
            avgq[i, m] = -(flattenSweepDict(inner_sweeps)["length"] + flattenSweepDict(inner_sweeps)["phase"]) * (
                        m + 1) + i

        bufi[i] = avgi[i].transpose().flatten() + (np.random.rand(reps, n_msmts * len(x1_pts) * len(x2_pts)) - 0.5) * 10
        bufq[i] = avgq[i].transpose().flatten() + (np.random.rand(reps, n_msmts * len(x1_pts) * len(x2_pts)) - 0.5) * 10

    outer_sweeps = DataDictBase(amp={"unit": "dBm", "values": np.linspace(-20, -5, 16)},
                                freq={"unit": "MHz", "values": np.linspace(1000, 1005, 6)}
                                )

    qdd = QickDataDict(ro_chs, inner_sweeps, outer_sweeps)
    # qdd.add_data(x_pts, avgi, avgq)
    qdd.add_data(avgi, avgq, bufi, bufq, inner_sweeps, amp=1, freq=3)
    qdd.add_data(avgi, avgq, bufi, bufq, inner_sweeps, amp=1, freq=4)
    qdd.add_data(avgi, avgq, bufi, bufq, inner_sweeps, amp=2, freq=3)
    qdd.add_data(avgi, avgq, bufi, bufq, inner_sweeps, amp=2, freq=4)

    # the automatic griding in plottr doesn't work well in this complicated multidimensional sweep data.
    # We have to manually set the grid in the app.
    ap = autoplot(qdd)