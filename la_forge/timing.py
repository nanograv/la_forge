import corner

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import astropy.units as u

from copy import deepcopy
from collections import defaultdict
from collections import OrderedDict

from pint.residuals import Residuals

from .diagnostics import plot_chains
from .core import Core
from .core import TimingCore
from .utils import get_param_groups
from .utils import get_fancy_labels


def get_pardict(psrs, datareleases):
    """assigns a parameter dictionary for each psr per dataset the parfile values/errors

    :param psrs: enterprise pulsar instances corresponding to datareleases
    :param datareleases: list of datareleases
    """
    pardict = {}
    for psr, dataset in zip(psrs, datareleases):
        pardict[psr.name] = {}
        pardict[psr.name][dataset] = {}
        for par, vals, errs in zip(
            psr.fitpars[1:],
            np.longdouble(psr.t2pulsar.vals()),
            np.longdouble(psr.t2pulsar.errs()),
        ):
            pardict[psr.name][dataset][par] = {}
            pardict[psr.name][dataset][par]["val"] = vals
            pardict[psr.name][dataset][par]["err"] = errs
    return pardict


def make_dmx_file(parfile):
    """Strips the parfile for the dmx values to be used in an Advanced Noise Modeling Run

    :param parfile: the parameter file to be stripped
    """
    dmx_dict = {}
    with open(parfile, "r") as f:
        lines = f.readlines()

    for line in lines:
        splt_line = line.split()
        if "DMX" in splt_line[0] and splt_line[0] != "DMX":
            for dmx_group in [
                y.split()
                for y in lines
                if str(splt_line[0].split("_")[-1]) in str(y.split()[0])
            ]:
                # Columns: DMXEP DMX_value DMX_var_err DMXR1 DMXR2 DMXF1 DMXF2 DMX_bin
                lab = f"DMX_{dmx_group[0].split('_')[-1]}"
                if lab not in dmx_dict.keys():
                    dmx_dict[lab] = {}
                if "DMX_" in dmx_group[0]:
                    if isinstance(dmx_group[1], str):
                        dmx_dict[lab]["DMX_value"] = np.double(
                            ("e").join(dmx_group[1].split("D"))
                        )
                    else:
                        dmx_dict[lab]["DMX_value"] = np.double(dmx_group[1])
                    if isinstance(dmx_group[-1], str):
                        dmx_dict[lab]["DMX_var_err"] = np.double(
                            ("e").join(dmx_group[-1].split("D"))
                        )
                    else:
                        dmx_dict[lab]["DMX_var_err"] = np.double(dmx_group[-1])
                    dmx_dict[lab]["DMX_bin"] = "DX" + dmx_group[0].split("_")[-1]
                else:
                    dmx_dict[lab][dmx_group[0].split("_")[0]] = np.double(dmx_group[1])
    for dmx_name, dmx_attrs in dmx_dict.items():
        if any([key for key in dmx_attrs.keys() if "DMXEP" not in key]):
            dmx_dict[dmx_name]["DMXEP"] = (
                dmx_attrs["DMXR1"] + dmx_attrs["DMXR2"]
            ) / 2.0
    dmx_df = pd.DataFrame.from_dict(dmx_dict, orient="index")
    neworder = [
        "DMXEP",
        "DMX_value",
        "DMX_var_err",
        "DMXR1",
        "DMXR2",
        "DMXF1",
        "DMXF2",
        "DMX_bin",
    ]
    final_order = []
    for order in neworder:
        if order in dmx_dict["DMX_0001"]:
            final_order.append(order)
    dmx_df = dmx_df.reindex(columns=final_order)
    new_dmx_file = (".dmx").join(parfile.split(".par"))
    with open(new_dmx_file, "w") as f:
        f.write(
            f"# {parfile.split('/')[-1].split('.par')[0]} dispersion measure variation\n"
        )
        f.write(
            f"# Mean DMX value = {np.mean([dmx_dict[x]['DMX_value'] for x in dmx_dict.keys()])} \n"
        )
        f.write(
            f"# Uncertainty in average DM = {np.std([dmx_dict[x]['DMX_value'] for x in dmx_dict.keys()])} \n"
        )
        f.write(f"# Columns: {(' ').join(final_order)}\n")
        dmx_df.to_csv(f, sep=" ", index=False, header=False)


def tm_delay(psr, tm_params_orig, new_params, plot=True):
    """
    Compute difference in residuals due to perturbed timing model.

    :param psr: enterprise pulsar object
    :param tm_params_orig: dictionary of TM parameter tuples, (val, err)
    :param new_params: dictionary of new TM parameter tuples, (val, err)
    :param plot: Whether to plot the delay or return the delay
    """
    if hasattr(psr, "model"):
        residuals = Residuals(psr.pint_toas, psr.model)
    elif hasattr(psr, "t2pulsar"):
        residuals = np.longdouble(psr.t2pulsar.residuals())
    else:
        raise ValueError(
            "Enterprise pulsar must keep either pint or t2pulsar. Use either drop_t2pulsar=False or drop_pintpsr=False when initializing the enterprise pulsar."
        )

    # grab original timing model parameters and errors in dictionary
    orig_params = {}
    tm_params_rescaled = {}
    for tm_scaled_key, tm_scaled_val in new_params.items():
        if "DMX" in tm_scaled_key:
            tm_param = "_".join(tm_scaled_key.split("_")[-2:])
        else:
            tm_param = tm_scaled_key.split("_")[-1]

        if tm_param == "COSI":
            orig_params["SINI"] = np.longdouble(tm_params_orig["SINI"][0])
        else:
            orig_params[tm_param] = np.longdouble(tm_params_orig[tm_param][0])

        if "physical" in tm_params_orig[tm_param]:
            # User defined priors are assumed to not be scaled
            if tm_param == "COSI":
                # Switch for sampling in COSI, but using SINI in libstempo
                tm_params_rescaled["SINI"] = np.longdouble(
                    np.sqrt(1 - tm_scaled_val**2)
                )
            else:
                tm_params_rescaled[tm_param] = np.longdouble(tm_scaled_val)
        else:
            if tm_param == "COSI":
                # Switch for sampling in COSI, but using SINI in libstempo
                rescaled_COSI = np.longdouble(
                    tm_scaled_val * tm_params_orig[tm_param][1]
                    + tm_params_orig[tm_param][0]
                )
                tm_params_rescaled["SINI"] = np.longdouble(
                    np.sqrt(1 - rescaled_COSI**2)
                )
                # print("Rescaled COSI used to find SINI", np.longdouble(rescaled_COSI))
                # print("rescaled SINI", tm_params_rescaled["SINI"])
            else:
                tm_params_rescaled[tm_param] = np.longdouble(
                    tm_scaled_val * tm_params_orig[tm_param][1]
                    + tm_params_orig[tm_param][0]
                )

    if hasattr(psr, "model"):
        new_model = deepcopy(psr.model)
        # Set values to new sampled values
        new_model.set_param_values(tm_params_rescaled)
        # Get new residuals
        # new_res = np.longdouble(Residuals(psr.pint_toas, new_model).resids_value)
        new_res = Residuals(psr.pint_toas, new_model)
        if plot:
            plotres_pint(psr, new_res, residuals)
    elif hasattr(psr, "t2pulsar"):
        # Set values to new sampled values
        psr.t2pulsar.vals(tm_params_rescaled)
        # Get new residuals
        new_res = np.longdouble(psr.t2pulsar.residuals())
        # Set values back to originals
        psr.t2pulsar.vals(orig_params)
        if plot:
            plotres_t2(psr, new_res)
    else:
        raise ValueError(
            "Enterprise pulsar must keep either pint or t2pulsar. Use either drop_t2pulsar=False or drop_pintpsr=False when initializing the enterprise pulsar."
        )

    if not plot:
        # Return the time-series for the pulsar
        return new_res[psr.isort], psr.residuals


def plotres_pint(psr, new_res, old_res, **kwargs):
    """Used to compare different sets of residuals from a pint pulsar
    :param psr: an `enterprise` pulsar with the `PINT` pulsar object retained
    :param new_res: the new residuals, post-run
    :param old_res: the old residuals, generally from the parfile parameters
    """
    kwargs.get("alpha", 0.5)
    plt.errorbar(
        old_res.toas.get_mjds().value,
        old_res.resids.to(u.us).value,
        yerr=old_res.toas.get_errors().to(u.us).value,
        fmt="x",
        label="Old Residuals",
        **kwargs,
    )
    plt.errorbar(
        new_res.toas.get_mjds().value,
        new_res.resids.to(u.us).value,
        yerr=new_res.toas.get_errors().to(u.us).value,
        fmt="+",
        label="New Residuals",
        **kwargs,
    )
    meannewres = np.sqrt(np.mean(new_res.resids.to(u.us) ** 2))
    meanoldres = np.sqrt(np.mean(old_res.resids.to(u.us) ** 2))

    plt.legend()
    plt.xlabel(r"MJD")
    plt.ylabel(r"res [$\mu s$]")
    plt.title(
        rf"{psr.name}: RMS, Old Res = {meanoldres.value:.3f} $\mu s$, New Res = {meannewres.value:.3f} $\mu s$"
    )
    plt.grid()


def plotres_t2(psr, new_res, **kwargs):
    """Used to compare different sets of residuals from a T2 pulsar
    :param psr: an `enterprise` pulsar with the `TEMPO2` pulsar object retained
    :param new_res: the new residuals, post-run
    """
    kwargs.get("alpha", 0.5)
    plt.errorbar(
        (psr.toas * u.s).to("d").value,
        (psr.residuals * u.s).to("us").value,
        yerr=(psr.toaerrs * u.s).to("us").value,
        fmt="x",
        label="Old Residuals",
        **kwargs,
    )
    plt.errorbar(
        (psr.toas * u.s).to("d").value,
        (new_res[psr.isort] * u.s).to("us").value,
        yerr=(psr.toaerrs * u.s).to("us").value,
        fmt="+",
        label="New Residuals",
        **kwargs,
    )
    meannewres = np.sqrt(np.mean((new_res * u.s).to("us") ** 2))
    meanoldres = np.sqrt(np.mean((psr.residuals * u.s).to("us") ** 2))

    plt.legend()
    plt.xlabel(r"MJD")
    plt.ylabel(r"res [$\mu s$]")
    plt.title(
        rf"{psr.name}: RMS, Old Res = {meanoldres.value:.3f} $\mu s$, New Res = {meannewres.value:.3f} $\mu s$"
    )
    plt.grid()


def residual_comparison(
    psr,
    core,
    use_mean_median_map="median",
    close=False
):
    """Used to compare old residuals to new residuals.

    :param psr: `enterprise` pulsar object with either `PINT` or `libstempo` pulsar retained
    :param core: `la_forge` core object
    :param use_mean_median_map: {"median","mean","map"} determines which values from posteriors to take
        as values in the residual calculation
    """
    core_titles = get_titles(psr.name, core)
    core_timing_dict_unscaled = OrderedDict()

    if use_mean_median_map == "map":
        map_idx_e_e = np.argmax(core.get_param("lnpost", to_burn=True))
    elif use_mean_median_map not in ["map", "median", "mean"]:
        raise ValueError(
            "use_mean_median_map can only be either 'map','median', or 'mean"
        )

    for par in core.params:
        unscaled_param = core.get_param(par, to_burn=True, tm_convert=True)
        if use_mean_median_map == "map":
            core_timing_dict_unscaled[par] = unscaled_param[map_idx_e_e]
        elif use_mean_median_map == "mean":
            core_timing_dict_unscaled[par] = np.mean(unscaled_param)
        elif use_mean_median_map == "median":
            core_timing_dict_unscaled[par] = np.median(unscaled_param)

    core_timing_dict = deepcopy(core_timing_dict_unscaled)
    """
    if use_tm_pars_orig:
        for p in core_timing_dict.keys():
            if "timing" in p:
                core_timing_dict.update(
                    {p: np.double(core.tm_pars_orig[p.split("_")[-1]][0])}
                )
    """
    chain_tm_params_orig = deepcopy(core.tm_pars_orig)
    chain_tm_delay_kwargs = {}
    for par in psr.fitpars:
        if par == "SINI" and "COSI" in core.tm_pars_orig.keys():
            sin_val, sin_err, _ = chain_tm_params_orig[par]
            val = np.longdouble(np.sqrt(1 - sin_val**2))
            err = np.longdouble(np.sqrt((np.abs(sin_val / val)) ** 2 * sin_err**2))
            chain_tm_params_orig["COSI"] = [val, err, "physical"]
            chain_tm_delay_kwargs["COSI"] = core_timing_dict[
                core.params[core_titles.index("COSI")]
            ]
        elif par in core.tm_pars_orig.keys():
            if par in core_titles:
                chain_tm_params_orig[par][-1] = "physical"
                chain_tm_delay_kwargs[par] = core_timing_dict[
                    core.params[core_titles.index(par)]
                ]
            else:
                # print(f"{par} not directly sampled. Using parfile value.")
                chain_tm_params_orig[par][-1] = "normalized"
                chain_tm_delay_kwargs[par] = 0.0
        else:
            print(f"{par} not in psr pars")

    tm_delay(psr, chain_tm_params_orig, chain_tm_delay_kwargs)
    if close:
        plt.close()
    else:
        plt.show()


def summary_comparison(psr_name, core, selection="all"):
    """Makes comparison table of the form:
    Par Name | Old Value | New Value | Difference | Old Sigma | New Sigma
    :param psr_name: str, Name of the pulsar to be looked at
    :param core: `la_forge` core object
    :param selection: str, Used to select various groups of parameters:
        see `get_param_groups` for details
    """
    # TODO: allow for selection of subparameters
    pd.set_option("display.max_rows", None)
    plot_params = get_param_groups(core, selection=selection)
    summary_dict = {}
    for pnames, title in zip(plot_params["par"], plot_params["title"]):
        if "timing" in pnames:
            if isinstance(core, TimingCore):
                param_vals = core.get_param(pnames, tm_convert=True, to_burn=True)
            elif isinstance(core, Core):
                param_vals = core.get_param(pnames, to_burn=True)

            summary_dict[title] = {}
            summary_dict[title]["new_val"] = np.median(param_vals)
            # summary_dict[title]["new_sigma"] = np.std(param_vals)
            summary_dict[title]["new_sigma"] = np.max(
                np.abs(
                    core.get_param_credint(pnames, interval=68, onesided=False)
                    - summary_dict[title]["new_val"]
                )
            )
            summary_dict[title]["rounded_new_sigma"] = np.round(
                summary_dict[title]["new_sigma"],
                -int(np.floor(np.log10(np.abs(summary_dict[title]["new_sigma"])))),
            )
            summary_dict[title]["rounded_new_val"] = np.round(
                summary_dict[title]["new_val"],
                -int(np.floor(np.log10(np.abs(summary_dict[title]["new_sigma"])))),
            )

            if title in core.tm_pars_orig:
                summary_dict[title]["old_val"] = core.tm_pars_orig[title][0]
                summary_dict[title]["old_sigma"] = core.tm_pars_orig[title][1]
                summary_dict[title]["rounded_old_sigma"] = np.round(
                    summary_dict[title]["old_sigma"],
                    -int(np.floor(np.log10(np.abs(summary_dict[title]["old_sigma"])))),
                )
                summary_dict[title]["rounded_old_val"] = np.round(
                    summary_dict[title]["old_val"],
                    -int(np.floor(np.log10(np.abs(summary_dict[title]["old_sigma"])))),
                )
                summary_dict[title]["difference"] = (
                    summary_dict[title]["new_val"] - core.tm_pars_orig[title][0]
                )
                if abs(summary_dict[title]["difference"]) > core.tm_pars_orig[title][1]:
                    summary_dict[title]["big"] = True
                else:
                    summary_dict[title]["big"] = False
                if summary_dict[title]["new_sigma"] < summary_dict[title]["old_sigma"]:
                    summary_dict[title]["constrained"] = True
                else:
                    summary_dict[title]["constrained"] = False
            else:
                summary_dict[title]["old_val"] = "-"
                summary_dict[title]["old_sigma"] = "-"
                summary_dict[title]["difference"] = "-"
                summary_dict[title]["big"] = "-"
                summary_dict[title]["constrained"] = "-"
    return pd.DataFrame(
        np.asarray(
            [
                [x for x in summary_dict.keys()],
                [summary_dict[x]["old_val"] for x in summary_dict.keys()],
                [summary_dict[x]["new_val"] for x in summary_dict.keys()],
                [summary_dict[x]["difference"] for x in summary_dict.keys()],
                [summary_dict[x]["old_sigma"] for x in summary_dict.keys()],
                [summary_dict[x]["new_sigma"] for x in summary_dict.keys()],
                [summary_dict[x]["rounded_old_val"] for x in summary_dict.keys()],
                [summary_dict[x]["rounded_old_sigma"] for x in summary_dict.keys()],
                [summary_dict[x]["rounded_new_val"] for x in summary_dict.keys()],
                [summary_dict[x]["rounded_new_sigma"] for x in summary_dict.keys()],
                [summary_dict[x]["big"] for x in summary_dict.keys()],
                [summary_dict[x]["constrained"] for x in summary_dict.keys()],
            ]
        ).T,
        columns=[
            "Parameter",
            "Old Value",
            "New Median Value",
            "Difference",
            "Old Sigma",
            "New Sigma",
            "Rounded Old Value",
            "Rounded Old Sigma",
            "Rounded New Value",
            "Rounded New Sigma",
            ">1 sigma change?",
            "More Constrained?",
        ],
    )


def get_titles(psr_name, core):
    """Get titles for timing model parameters
    :param psr_name: Name of the pulsar
    :param core: `la_forge` core object
    """
    titles = []
    for core_param in core.params:
        if "timing" in core_param.split("_"):
            if "DMX" in core_param.split("_"):
                titles.append(("_").join(core_param.split("_")[-2:]))
            else:
                titles.append(core_param.split("_")[-1])
        else:
            if psr_name in core_param.split("_"):
                titles.append((" ").join(core_param.split("_")[1:]))
            else:
                titles.append(core_param)
    return titles


def get_common_params_titles(psr_name, core_list, exclude=True):
    """Renames gets common parameters and titles
    :param psr_name: Name of the pulsar
    :param core_list: list of `la_forge` core objects
    :param exclude: exclude ["lnpost","lnlike","chain_accept","pt_chain_accept",]
    """
    common_params = []
    common_titles = []
    for core in core_list:
        if len(common_params) == 0:
            for core_param in core.params:
                common_params.append(core_param)
                if "timing" in core_param.split("_"):
                    if "DMX" in core_param.split("_"):
                        common_titles.append(("_").join(core_param.split("_")[-2:]))
                    else:
                        common_titles.append(core_param.split("_")[-1])
                else:
                    if psr_name in core_param.split("_"):
                        common_titles.append((" ").join(core_param.split("_")[1:]))
                    else:
                        common_titles.append(core_param)
        else:
            if exclude:
                uncommon_params = [
                    "lnpost",
                    "lnlike",
                    "chain_accept",
                    "pt_chain_accept",
                ]
            else:
                uncommon_params = ["chain_accept", "pt_chain_accept"]

            for com_par in common_params:
                if com_par not in core.params:
                    uncommon_params.append(com_par)
            for ncom_par in uncommon_params:
                if ncom_par in common_params:
                    del common_titles[common_params.index(ncom_par)]
                    del common_params[common_params.index(ncom_par)]
    return common_params, common_titles


def get_other_param_overlap(psr_name, core_list):
    """Gets a dictionary of params with list of indices for corresponding core in core_list
    :param psr_name: Name of the pulsar
    :param core_list: list of `la_forge` core objects
    """
    boring_params = ["lnpost", "lnlike", "chain_accept", "pt_chain_accept"]
    common_params_all, _ = get_common_params_titles(psr_name, core_list, exclude=False)
    com_par_dict = defaultdict(list)
    for j, core in enumerate(core_list):
        for param in core.params:
            if param not in common_params_all + boring_params:
                com_par_dict[param].append(j)

    return com_par_dict


def plot_all_param_overlap(
    psr_name,
    core_list,
    core_list_legend=None,
    exclude=True,
    real_tm_pars=True,
    conf_int=None,
    close=True,
    show_plt=True,
    par_sigma={},
    ncols=3,
    hist_kwargs={},
    fig_kwargs={},
    **kwargs,
):
    """Plots common parameters between cores in core_list
    :param psr_name: Name of the pulsar
    :param core_list: list of `la_forge` core objects
    :param core_list_legend: list of labels corresponding to core_list
    :param exclude: excludes ["lnpost","lnlike","chain_accept","pt_chain_accept",]
    :param real_tm_pars: Whether to plot scaled or unscaled Timing Model parameters
    :param conf_int: float shades confidence interval region can be float between 0 and 1
    :param close: Whether to close the figure after displaying
    :param par_sigma: the error dictionary from the parfile of the form: {par_name:(val,err,'physical')}
    :param ncols: number of columns to plot
    :param hist_kwargs: kwargs for the histograms
    :param fig_kwargs: general figure kwargs
    """
    if not core_list_legend:
        core_list_legend = []
        for core in core_list:
            core_list_legend.append(core.label)

    linestyles = kwargs.get("linestyles", ["-" for x in core_list])
    suptitle = fig_kwargs.get("suptitle", f"{psr_name} Comparison Plots")
    # labelfontsize = fig_kwargs.get("labelfontsize", 18)
    # titlefontsize = fig_kwargs.get("titlefontsize", 16)
    suptitlefontsize = fig_kwargs.get("suptitlefontsize", 24)
    suptitleloc = fig_kwargs.get("suptitleloc", (0.35, 0.94))
    legendloc = fig_kwargs.get("legendloc", (0.45, 0.97))
    legendfontsize = fig_kwargs.get("legendfontsize", 12)
    colors = fig_kwargs.get("colors", [f"C{ii}" for ii in range(len(core_list_legend))])
    wspace = fig_kwargs.get("wspace", 0.1)
    hspace = fig_kwargs.get("hspace", 0.4)
    # figsize = fig_kwargs.get("figsize", (15, 10))

    hist_core_list_kwargs = {
        "hist": True,
        "ncols": ncols,
        "title_y": 1.4,
        "hist_kwargs": hist_kwargs,
        "linewidth": 3.0,
        "linestyle": linestyles,
    }

    common_params_all, common_titles_all = get_common_params_titles(
        psr_name, core_list, exclude=exclude
    )
    hist_core_list_kwargs["title_y"] = 1.0 + 0.025 * len(core_list)
    if len(core_list) < 3:
        y = 0.98 - 0.01 * len(core_list)
        hist_core_list_kwargs["title_y"] = 1.0 + 0.02 * len(core_list)
    elif len(core_list) >= 3 and len(core_list) < 5:
        y = 0.95 - 0.01 * len(core_list)
        # hist_core_list_kwargs['title_y'] = 1.+.05*len(core_list)
    elif len(core_list) >= 3 and len(core_list) < 10:
        y = 0.95 - 0.01 * len(core_list)
        # hist_core_list_kwargs['title_y'] = 1.+.025*len(core_list)
    else:
        y = 0.95 - 0.01 * len(core_list)
        # hist_core_list_kwargs['title_y'] = 1.+.05*len(core_list)

    if close:
        plot_chains(
            core_list,
            suptitle=suptitle,
            pars=common_params_all,
            titles=common_titles_all,
            real_tm_pars=real_tm_pars,
            show=False,
            close=False,
            **hist_core_list_kwargs,
        )

        if conf_int or par_sigma:
            fig = plt.gcf()
            allaxes = fig.get_axes()
            for ax in allaxes:
                splt_key = ax.get_title()
                if splt_key in par_sigma:
                    val = par_sigma[splt_key][0]
                    err = par_sigma[splt_key][1]
                    fill_space_x = np.linspace(val - err, val + err, 20)
                    ax.fill_between(
                        fill_space_x, ax.get_ylim()[1], color="grey", alpha=0.2
                    )
                    ax.axvline(val, color="k", linestyle="--")
                elif splt_key == "COSI" and "SINI" in par_sigma:
                    sin_val, sin_err, _ = par_sigma["SINI"]
                    val = np.longdouble(np.sqrt(1 - sin_val**2))
                    err = np.longdouble(
                        np.sqrt((np.abs(sin_val / val)) ** 2 * sin_err**2)
                    )
                    fill_space_x = np.linspace(val - err, val + err, 20)
                    ax.fill_between(
                        fill_space_x, ax.get_ylim()[1], color="grey", alpha=0.2
                    )
                    ax.axvline(val, color="k", linestyle="--")

                if conf_int:
                    if isinstance(conf_int, (float, int)):
                        if conf_int < 1.0 or conf_int > 99.0:
                            raise ValueError("conf_int must be between 1 and 99")
                    else:
                        raise ValueError("conf_int must be an int or float")

                    for i, core in enumerate(core_list):
                        # elif splt_key == 'COSI' and 'SINI' in par_sigma:
                        for com_par, com_title in zip(
                            common_params_all, common_titles_all
                        ):
                            if splt_key == com_title:
                                low, up = core.get_param_credint(
                                    com_par, interval=conf_int
                                )
                                ax.fill_between(
                                    [low, up],
                                    ax.get_ylim()[1],
                                    color=f"C{i}",
                                    alpha=0.1,
                                )
        else:
            fig = plt.gcf()
        patches = []
        for jj, lab in enumerate(core_list_legend):
            patches.append(
                mpl.patches.Patch(
                    color=colors[jj],
                    linestyle=linestyles[jj],
                    fill=False,
                    label=lab,
                    linewidth=3,
                )
            )  # .split(":")[-1]))

        fig.legend(handles=patches, loc=legendloc, fontsize=legendfontsize)
        fig.subplots_adjust(wspace=wspace, hspace=hspace)
        plt.suptitle(
            suptitle,
            fontsize=suptitlefontsize,
            x=suptitleloc[0],
            y=suptitleloc[1],
        )
        if show_plt:
            plt.show()
        plt.close()
    else:
        plot_chains(
            core_list,
            suptitle=psr_name,
            pars=common_params_all,
            titles=common_titles_all,
            legend_labels=core_list_legend,
            legend_loc=(0.0, y),
            real_tm_pars=real_tm_pars,
            **hist_core_list_kwargs,
        )


def plot_other_param_overlap(
    psr_name,
    core_list,
    core_list_legend=None,
    real_tm_pars=True,
    close=True,
    show_plt=True,
    selection="all",
    par_sigma={},
    hist_kwargs={},
    **kwargs,
):
    """Plots non-common parameters between cores in core_list
    :param psr_name: Name of the pulsar
    :param core_list: list of `la_forge` core objects
    :param core_list_legend: list of labels corresponding to core_list
    :param real_tm_pars: Whether to plot scaled or unscaled Timing Model parameters
    :param close: Whether to close the figure after displaying
    :param selection: str, Used to select various groups of parameters:
        see `get_param_groups` for details
    :param par_sigma: the error dictionary from the parfile of the form: {par_name:(val,err,'physical')}
    :param hist_kwargs: kwargs for the histograms
    """
    linestyles = kwargs.get("linestyles", ["-" for x in core_list])
    legendloc = kwargs.get("legendloc", (0.0, 0.95 - 0.03 * len(core_list)))
    title_y = kwargs.get("title_y", 1.0 + 0.05 * len(core_list))

    com_par_dict = get_other_param_overlap(psr_name, core_list)
    if not core_list_legend:
        core_list_legend = []
        for core in core_list:
            core_list_legend.append(core.label)

    hist_core_list_kwargs = {
        "hist": True,
        "ncols": 1,
        "title_y": title_y,
        "hist_kwargs": hist_kwargs,
        "linewidth": 3.0,
        "linestyle": linestyles,
    }

    selected_params = []
    for c_l in core_list:
        tmp_selected_params = get_param_groups(c_l, selection=selection)
        for tsp in tmp_selected_params["par"]:
            if tsp not in selected_params:
                selected_params.append(tsp)

    for key, vals in com_par_dict.items():
        if key in selected_params:
            if close:
                # hist_core_list_kwargs["title_y"] = 1.0 + 0.05 * len(core_list)
                plot_chains(
                    [core_list[c] for c in vals],
                    suptitle=psr_name,
                    pars=[key],
                    legend_labels=[core_list_legend[c] for c in vals],
                    legend_loc=legendloc,
                    real_tm_pars=real_tm_pars,
                    show=False,
                    close=False,
                    **hist_core_list_kwargs,
                )
                if par_sigma:
                    fig = plt.gcf()
                    allaxes = fig.get_axes()
                    for ax in allaxes:
                        full_key = ax.get_title()
                        splt_key = full_key.split("_")
                        if "DMX" in splt_key:
                            par_key = ("_").join(splt_key[-2:])
                        else:
                            par_key = splt_key[-1]
                        if par_key in par_sigma:
                            val = par_sigma[par_key][0]
                            err = par_sigma[par_key][1]
                            fill_space_x = np.linspace(val - err, val + err, 20)
                            ax.fill_between(
                                fill_space_x, ax.get_ylim()[1], color="grey", alpha=0.2
                            )
                            ax.axvline(val, color="k", linestyle="--")
                        elif splt_key == "COSI" and "SINI" in par_sigma:
                            sin_val, sin_err, _ = par_sigma["SINI"]
                            val = np.longdouble(np.sqrt(1 - sin_val**2))
                            err = np.longdouble(
                                np.sqrt((np.abs(sin_val / val)) ** 2 * sin_err**2)
                            )
                            fill_space_x = np.linspace(val - err, val + err, 20)
                            ax.fill_between(
                                fill_space_x, ax.get_ylim()[1], color="grey", alpha=0.2
                            )
                            ax.axvline(val, color="k", linestyle="--")
                if show_plt:
                    plt.show()
                plt.close()
            else:
                # hist_core_list_kwargs["title_y"] = 1.0 + 0.05 * len(core_list)
                plot_chains(
                    [core_list[c] for c in vals],
                    suptitle=psr_name,
                    pars=[key],
                    legend_labels=[core_list_legend[c] for c in vals],
                    legend_loc=legendloc,
                    real_tm_pars=real_tm_pars,
                    show=False,
                    close=False,
                    **hist_core_list_kwargs,
                )

                if par_sigma:
                    fig = plt.gcf()
                    allaxes = fig.get_axes()
                    for ax in allaxes:
                        splt_key = ax.get_title()
                        if splt_key in par_sigma:
                            val = par_sigma[splt_key][0]
                            err = par_sigma[splt_key][1]
                            fill_space_x = np.linspace(val - err, val + err, 20)
                            ax.fill_between(
                                fill_space_x, ax.get_ylim()[1], color="grey", alpha=0.2
                            )
                            ax.axvline(val, color="k", linestyle="--")
                        elif splt_key == "COSI" and "SINI" in par_sigma:
                            sin_val, sin_err, _ = par_sigma["SINI"]
                            val = np.longdouble(np.sqrt(1 - sin_val**2))
                            err = np.longdouble(
                                np.sqrt((np.abs(sin_val / val)) ** 2 * sin_err**2)
                            )
                            fill_space_x = np.linspace(val - err, val + err, 20)
                            ax.fill_between(
                                fill_space_x, ax.get_ylim()[1], color="grey", alpha=0.2
                            )
                            ax.axvline(val, color="k", linestyle="--")
            print("")


def fancy_plot_all_param_overlap(
    psr_name,
    core_list,
    core_list_legend=None,
    exclude=True,
    par_sigma={},
    conf_int=False,
    preliminary=True,
    ncols=4,
    real_tm_pars=True,
    show_plt=True,
    selection="all",
    hist_kwargs={},
    fig_kwargs={},
    **kwargs,
):
    """Plots common parameters between cores in core_list with fancier plotting methods
    :param psr_name: Name of the pulsar
    :param core_list: list of `la_forge` core objects
    :param core_list_legend: list of labels corresponding to core_list
    :param exclude: excludes ["lnpost","lnlike","chain_accept","pt_chain_accept",]
    :param par_sigma: the error dictionary from the parfile of the form: {par_name:(val,err,'physical')}
    :param conf_int: float shades confidence interval region can be float between 0 and 1
    :param preliminary: Whether to display large 'preliminary' over plot
    :param ncols: number of columns to plot
    :param real_tm_pars: Whether to plot scaled or unscaled Timing Model parameters
    :param selection: str, Used to select various groups of parameters:
        see `get_param_groups` for details
    :param hist_kwargs: kwargs for the histograms
    :param fig_kwargs: general figure kwargs
    """
    if not core_list_legend:
        core_list_legend = []
        for core in core_list:
            core_list_legend.append(core.label)

    if not hist_kwargs:
        hist_kwargs = {
            "linewidth": 3.0,
            "density": True,
            "histtype": "step",
            "bins": 40,
        }

    linestyles = kwargs.get("linestyles", ["-" for x in core_list])
    close = kwargs.get("close", True)
    show_suptitle = kwargs.get("show_suptitle", True)

    suptitle = fig_kwargs.get("suptitle", f"{psr_name} Comparison Plots")
    labelfontsize = fig_kwargs.get("labelfontsize", 18)
    titlefontsize = fig_kwargs.get("titlefontsize", 16)
    suptitlefontsize = fig_kwargs.get("suptitlefontsize", 24)
    suptitleloc = fig_kwargs.get("suptitleloc", (0.35, 0.94))
    legendloc = fig_kwargs.get("legendloc", (0.5, 0.93))
    legendfontsize = fig_kwargs.get("legendfontsize", 12)
    colors = fig_kwargs.get("colors", [f"C{ii}" for ii in range(len(core_list_legend))])
    wspace = fig_kwargs.get("wspace", 0.1)
    hspace = fig_kwargs.get("hspace", 0.4)
    # figsize = fig_kwargs.get("figsize", (15, 10))

    common_params_all, common_titles_all = get_common_params_titles(
        psr_name, core_list, exclude=exclude
    )

    fancy_labels = get_fancy_labels(common_titles_all)

    selected_params = get_param_groups(core_list[0], selection=selection)

    selected_common_params = []
    selected_common_titles = []
    selected_fancy_labels = []
    for cpa, cta, fl in zip(common_params_all, common_titles_all, fancy_labels):
        if cpa in selected_params["par"]:
            selected_common_params.append(cpa)
            selected_common_titles.append(cta)
            selected_fancy_labels.append(fl)

    if preliminary:
        txt = "PRELIMINARY"
        txt_loc = (0.15, 0.15)
        txt_kwargs = {"fontsize": 180, "alpha": 0.25, "rotation": 55}

    L = len(selected_common_params)
    nrows = int(L // ncols)
    if L % ncols > 0:
        nrows += 1
    fig = plt.figure(figsize=[15, 4 * nrows])
    for (ii, par), label in zip(
        enumerate(selected_common_params), selected_fancy_labels
    ):
        if par in selected_params["par"]:
            cell = ii + 1
            axis = fig.add_subplot(nrows, ncols, cell)
            for jj, co in enumerate(core_list):
                if isinstance(co, TimingCore):
                    plt.hist(
                        co.get_param(par, tm_convert=real_tm_pars),
                        linestyle=linestyles[jj],
                        **hist_kwargs,
                    )
                elif isinstance(co, Core):
                    plt.hist(co.get_param(par), linestyle=linestyles[jj], **hist_kwargs)
            if "efac" in selected_common_titles[ii]:
                axis.set_title(
                    selected_common_titles[ii].split("efac")[0], fontsize=titlefontsize
                )
                axis.set_xlabel(label, fontsize=labelfontsize - 2)
            elif "equad" in selected_common_titles[ii]:
                axis.set_title(
                    selected_common_titles[ii].split("log10")[0], fontsize=titlefontsize
                )
                axis.set_xlabel(label, fontsize=labelfontsize - 2)
            elif "ecorr" in selected_common_titles[ii]:
                axis.set_title(
                    selected_common_titles[ii].split("log10")[0], fontsize=titlefontsize
                )
                axis.set_xlabel(label, fontsize=labelfontsize - 2)
            else:
                axis.set_xlabel(label, fontsize=labelfontsize)

            if "DMX" in par:
                splt_key = ("_").join(par.split("_")[-2:])
            else:
                splt_key = par.split("_")[-1]
            if par_sigma:
                if splt_key in par_sigma:
                    val = par_sigma[splt_key][0]
                    err = par_sigma[splt_key][1]
                    fill_space_x = np.linspace(val - err, val + err, 20)
                    axis.fill_between(
                        fill_space_x, axis.get_ylim()[1], color="grey", alpha=0.2
                    )
                    axis.axvline(val, color="k", linestyle="--")
                elif splt_key == "COSI" and "SINI" in par_sigma:
                    sin_val, sin_err, _ = par_sigma["SINI"]
                    val = np.longdouble(np.sqrt(1 - sin_val**2))
                    err = np.longdouble(
                        np.sqrt((np.abs(sin_val / val)) ** 2 * sin_err**2)
                    )
                    fill_space_x = np.linspace(val - err, val + err, 20)
                    axis.fill_between(
                        fill_space_x, axis.get_ylim()[1], color="grey", alpha=0.2
                    )
                    axis.axvline(val, color="k", linestyle="--")

            if conf_int:
                if isinstance(conf_int, (float, int)):
                    if conf_int < 1.0 or conf_int > 99.0:
                        raise ValueError("conf_int must be between 1 and 99")
                else:
                    raise ValueError("conf_int must be an int or float")

                for i, core in enumerate(core_list):
                    # elif splt_key == 'COSI' and 'SINI' in par_sigma:
                    for com_par, com_title in zip(
                        selected_common_params, selected_common_titles
                    ):
                        if splt_key == com_title:
                            low, up = core.get_param_credint(com_par, interval=conf_int)
                            axis.fill_between(
                                [low, up],
                                axis.get_ylim()[1],
                                color=f"C{i}",
                                alpha=0.1,
                            )
        axis.set_yticks([])
    patches = []
    for jj, lab in enumerate(core_list_legend):
        patches.append(
            mpl.patches.Patch(
                color=colors[jj],
                linestyle=linestyles[jj],
                fill=False,
                label=lab,
                linewidth=3,
            )
        )

    if preliminary:
        fig.text(txt_loc[0], txt_loc[1], txt, **txt_kwargs)
    fig.legend(handles=patches, loc=legendloc, fontsize=legendfontsize)
    fig.subplots_adjust(wspace=wspace, hspace=hspace)
    # fig.subplots_adjust(top=0.96)

    if show_suptitle:
        plt.suptitle(
            suptitle,
            fontsize=suptitlefontsize,
            x=suptitleloc[0],
            y=suptitleloc[1],
        )
    # plt.savefig(f'Figures/{psr_name}_cfr19_common_pars_2.png', dpi=150, bbox_inches='tight')
    # plt.savefig(f'Figures/{psr_name}_12p5yr_common_pars.png', dpi=150, bbox_inches='tight')
    if close:
        if show_plt:
            plt.show()
        plt.close()


def corner_plots(
    psr_name,
    core,
    selection="kep",
    save=False,
    real_tm_pars=False,
    truths=True,
    show_plt=True,
    hist2d_kwargs={},
    corner_label_kwargs={},
):
    """Plots a corner plot for core
    :param psr_name: Name of the pulsar
    :param core: `la_forge` core object
    :param save: Whether to save the figure
    :param selection: str, Used to select various groups of parameters:
        see `get_param_groups` for details
    :param real_tm_pars: Whether to plot scaled or unscaled Timing Model parameters
    :param truths: Whether to plot shaded truth regions (only assumes scaled for now)
    :param hist2d_kwargs: kwargs for the histograms
    :param corner_label_kwargs: kwargs for the corner labels
    """
    if not hist2d_kwargs:
        hist2d_kwargs = {
            "plot_density": False,
            "no_fill_contours": True,
            "data_kwargs": {"alpha": 0.02},
        }
    if not corner_label_kwargs:
        corner_label_kwargs = {"fontsize": 20, "rotation": 0}

    plot_params = get_param_groups(core, selection=selection)

    if truths:
        plot_truths = defaultdict(list)
        for pp in plot_params["par"]:
            if "timing" in pp:
                val, err, normed = core.tm_pars_orig[pp.split("_")[-1]]
                if normed == "normalized":
                    plot_truths["val"].append(0.0)
                    plot_truths["min"].append(-1.0)
                    plot_truths["max"].append(1.0)
                elif pp.split("_")[-1] in ["XDOT", "PBDOT"]:
                    plot_truths["val"].append(val)
                    if np.log10(err) > -10.0:
                        plot_truths["min"].append(val - err * 1e-12)
                        plot_truths["max"].append(val + err * 1e-12)
                    else:
                        plot_truths["min"].append(val - err)
                        plot_truths["max"].append(val + err)
                else:
                    plot_truths["val"].append(val)
                    plot_truths["min"].append(val - err)
                    plot_truths["max"].append(val + err)
            else:
                plot_truths["val"].append(-40.0)
                plot_truths["min"].append(-50.0)
                plot_truths["max"].append(-30.0)
    else:
        plot_truths = None

    # hist_kwargs = {"range": (-5, 5)}
    ranges = np.ones(len(plot_params)) * 0.95
    if isinstance(core, TimingCore):
        tmp_plt_params = []
        for ppar in plot_params["par"]:
            tmp_plt_params.append(core.get_param(ppar, tm_convert=real_tm_pars))
        plt_param = np.asarray(tmp_plt_params).T
    elif isinstance(core, Core):
        tmp_plt_params = []
        for ppar in plot_params["par"]:
            tmp_plt_params.append(core.get_param(ppar))
        plt_param = np.asarray(tmp_plt_params).T

    if truths:
        fig = corner.corner(
            plt_param,
            truths=plot_truths["val"],
            truth_color="k",
            color="C0",
            ranges=ranges,
            labels=plot_params["title"],
            levels=[0.68, 0.95],
            label_kwargs=corner_label_kwargs,
            labelpad=0.25,
            **hist2d_kwargs,
        )
    else:
        fig = corner.corner(
            plt_param,
            color="C0",
            # ranges=ranges,
            labels=plot_params["title"],
            levels=[0.68, 0.95],
            label_kwargs=corner_label_kwargs,
            labelpad=0.25,
            **hist2d_kwargs,
        )

    ndim = len(plot_params["par"])
    axes = np.array(fig.axes).reshape((ndim, ndim))

    if truths:
        # Loop over the histograms
        for yi in range(ndim):
            for xi in range(yi):
                ax = axes[yi, xi]
                if plot_truths["min"][xi] == -50:
                    fill_space_x = np.linspace(ax.get_xlim()[0], ax.get_xlim()[1], 20)
                else:
                    fill_space_x = np.linspace(
                        plot_truths["min"][xi], plot_truths["max"][xi], 20
                    )

                if plot_truths["min"][yi] == -50:
                    ax.fill_between(
                        fill_space_x,
                        ax.get_ylim()[0],
                        ax.get_ylim()[1],
                        color="y",
                        alpha=0.4,
                    )
                else:
                    ax.fill_between(
                        fill_space_x,
                        plot_truths["min"][yi],
                        plot_truths["max"][yi],
                        color="y",
                        alpha=0.4,
                    )

    fig.suptitle(f"{psr_name}, {selection} Parameters", fontsize=24)
    if save:
        plt.savefig(
            f"Figures/{selection}_Corner_{core.label}.png", dpi=150, bbox_inches="tight"
        )
    if show_plt:
        plt.show()


def mass_pulsar(PB, A1, SINI, M2, errors={}):
    """
    Computes the companion mass from the Keplerian mass function. This
    function uses a Newton-Raphson method since the equation is
    transcendental.

    :param PB: orbital period [days]
    :param A1: projected semimajor axis [lt-s]
    :param SINI: sine of the system inclination [degrees]
    :param M2: compaion mass [solar mass]
    :param errors: dictionary of errors on each param

    :return: pulsar mass [solar mass]
    """
    T_sun = 4.925490947e-6  # conversion from solar masses to seconds
    nb = 2 * np.pi / PB / 86400
    mf = nb**2 * A1**3 / T_sun

    if errors:
        mp_err_sqrd = (
            ((3 / 2) * np.sqrt(M2 * SINI**3 / mf) - 1) ** 2 * errors["M2"] ** 2
            + (((3 / 2) * np.sqrt(M2**3 * SINI / mf)) ** 2 * errors["SINI"] ** 2)
            + (
                (np.sqrt(M2**3 * SINI**3 / (2 * np.pi) ** 2)) ** 2
                * (errors["PB"] / 8600) ** 2
            )
            + (
                ((3 / 2) * np.sqrt(M2**2 * SINI / nb**2 / T_sun / A1)) ** 2
                * errors["A1"] ** 2
            )
        )

        return np.sqrt((M2 * SINI) ** 3 / mf) - M2, np.sqrt(mp_err_sqrd)
    else:
        return np.sqrt((M2 * SINI) ** 3 / mf) - M2


def mass_plot(
    psr_name,
    core_list,
    core_list_legend=None,
    real_tm_pars=True,
    preliminary=False,
    conf_int=None,
    print_conf_int=False,
    show_plt=True,
    close=False,
    par_sigma={},
    hist_kwargs={},
    fig_kwargs={},
    **kwargs,
):
    """Plots mass parameters for all cores in core_list
    :param psr_name: Name of the pulsar
    :param core_list: list of `la_forge` core objects
    :param core_list_legend: list of labels corresponding to core_list
    :param real_tm_pars: Whether to plot scaled or unscaled Timing Model parameters
    :param preliminary: Whether to display large 'preliminary' over plot
    :param conf_int: float shades confidence interval region can be float between 0 and 1
    :param print_conf_int: Whether to print out confidence intervals
    :param par_sigma: the error dictionary from the parfile of the form: {par_name:(val,err,'physical')}
    :param ncols: number of columns to plot
    :param hist_kwargs: kwargs for the histograms
    :param fig_kwargs: general figure kwargs
    """
    if not core_list_legend:
        core_list_legend = []
        for core in core_list:
            core_list_legend.append(core.label)

    hist_kwargs.update(
        {
            "linewidth": hist_kwargs.get("linewidth", 3.0),
            "density": hist_kwargs.get("density", True),
            "histtype": hist_kwargs.get("histtype", "step"),
            "bins": hist_kwargs.get("bins", 40),
        }
    )

    linestyles = kwargs.get("linestyles", ["-" for x in core_list])
    show_legend = kwargs.get("show_legend", True)
    show_suptitle = kwargs.get("show_suptitle", True)
    show_xlabel = kwargs.get("show_xlabel", True)
    xlabel_rotation = kwargs.get("xlabel_rotation", 0)

    suptitle = fig_kwargs.get("suptitle", f"{psr_name} Mass Plots")
    suptitlefontsize = fig_kwargs.get("suptitlefontsize", 24)
    suptitleloc = fig_kwargs.get("suptitleloc", (0.25, 1.01))
    legendloc = fig_kwargs.get("legendloc", (0.45, 0.925))
    legendfontsize = fig_kwargs.get("legendfontsize", 12)
    colors = fig_kwargs.get("colors", [f"C{ii}" for ii in range(len(core_list_legend))])
    wspace = fig_kwargs.get("wspace", 0.1)
    hspace = fig_kwargs.get("hspace", 0.4)
    figsize = fig_kwargs.get("figsize", (15, 10))

    if "fig" not in kwargs.keys() and "axes" not in kwargs.keys():
        fig, axes = plt.subplots(3, 1, figsize=figsize)
    else:
        fig = kwargs["fig"]
        axes = kwargs["axes"]

    co_labels = [
        r"Pulsar Mass$~(\mathrm{M}_{\odot})$",
        r"Companion Mass$~(\mathrm{M}_{\odot})$",
        r"$\mathrm{cos}~i$",
    ]
    for i, coco in enumerate(core_list):
        if isinstance(coco, TimingCore):
            co_Mc = coco.get_param(
                f"{psr_name}_timing_model_M2", to_burn=True, tm_convert=True
            )
            if f"{psr_name}_timing_model_COSI" in coco.params:
                co_COSI = coco.get_param(
                    f"{psr_name}_timing_model_COSI", to_burn=True, tm_convert=True
                )
                co_Mp = mass_pulsar(
                    coco.get_param(
                        f"{psr_name}_timing_model_PB", to_burn=True, tm_convert=True
                    ),
                    coco.get_param(
                        f"{psr_name}_timing_model_A1", to_burn=True, tm_convert=True
                    ),
                    np.sqrt((1 - co_COSI**2)),
                    coco.get_param(
                        f"{psr_name}_timing_model_M2", to_burn=True, tm_convert=True
                    ),
                )
            else:
                co_Mp = mass_pulsar(
                    coco.get_param(
                        f"{psr_name}_timing_model_PB", to_burn=True, tm_convert=True
                    ),
                    coco.get_param(
                        f"{psr_name}_timing_model_A1", to_burn=True, tm_convert=True
                    ),
                    coco.get_param(
                        f"{psr_name}_timing_model_SINI", to_burn=True, tm_convert=True
                    ),
                    coco.get_param(
                        f"{psr_name}_timing_model_M2", to_burn=True, tm_convert=True
                    ),
                )
                co_COSI = np.sqrt(
                    1
                    - coco.get_param(
                        f"{psr_name}_timing_model_SINI", to_burn=True, tm_convert=True
                    )
                    ** 2
                )
        elif isinstance(coco, Core):
            co_Mc = coco.get_param(f"{psr_name}_timing_model_M2", to_burn=True)
            if f"{psr_name}_timing_model_COSI" in coco.params:
                co_COSI = coco.get_param(f"{psr_name}_timing_model_COSI", to_burn=True)
                co_Mp = mass_pulsar(
                    coco.get_param(f"{psr_name}_timing_model_PB", to_burn=True),
                    coco.get_param(f"{psr_name}_timing_model_A1", to_burn=True),
                    np.sqrt((1 - co_COSI**2)),
                    coco.get_param(f"{psr_name}_timing_model_M2", to_burn=True),
                )
            else:
                co_Mp = mass_pulsar(
                    coco.get_param(f"{psr_name}_timing_model_PB", to_burn=True),
                    coco.get_param(f"{psr_name}_timing_model_A1", to_burn=True),
                    coco.get_param(f"{psr_name}_timing_model_SINI", to_burn=True),
                    coco.get_param(f"{psr_name}_timing_model_M2", to_burn=True),
                )
                co_COSI = co_COSI = np.sqrt(
                    1
                    - coco.get_param(f"{psr_name}_timing_model_SINI", to_burn=True) ** 2
                )
        co_bins = [co_Mp, co_Mc, co_COSI]

        if print_conf_int:
            print(coco.label)
            print("----------------")
        for j, ax in enumerate(axes):
            ax.hist(
                co_bins[j],
                label=core_list_legend[i],
                linestyle=linestyles[i],
                color=colors[i],
                **hist_kwargs,
            )
            if show_xlabel:
                ax.set_xlabel(co_labels[j], fontsize=24)
                ax.tick_params(axis="x", labelsize=16, rotation=xlabel_rotation)

            else:
                ax.set_xticklabels([])
            ax.get_yaxis().set_visible(False)

            if conf_int:
                if isinstance(conf_int, (float, int)):
                    if conf_int < 1.0 or conf_int > 99.0:
                        raise ValueError("conf_int must be between 1 and 99")
                else:
                    raise ValueError("conf_int must be an int or float")
                lower_q = (100 - conf_int) / 2
                lower = np.percentile(co_bins[j], q=lower_q)
                upper = np.percentile(co_bins[j], q=100 - lower_q)
                ax.fill_between(
                    [lower, upper], ax.get_ylim()[1], color=f"C{i}", alpha=0.1
                )

                if print_conf_int:
                    print(co_labels[j])
                    if j == 2:
                        med_val = np.arccos(np.median(co_bins[j])) * 180 / np.pi
                        diff_lower = (
                            (np.arccos(np.median(co_bins[j])) - np.arccos(upper))
                            * 180
                            / np.pi
                        )
                        diff_upper = (
                            (np.arccos(lower) - np.arccos(np.median(co_bins[j])))
                            * 180
                            / np.pi
                        )
                        print(f"Median: {np.arccos(np.median(co_bins[j]))*180/np.pi}")
                        print(f"Lower: {np.arccos(upper)*180/np.pi}")
                        print(f"Upper: {np.arccos(lower)*180/np.pi}")
                        print(
                            f"Diff Lower: {(np.arccos(np.median(co_bins[j]))-np.arccos(upper))*180/np.pi}"
                        )
                        print(
                            f"Diff Upper: {(np.arccos(lower)-np.arccos(np.median(co_bins[j])))*180/np.pi}"
                        )
                        print(
                            f"Rounded Median: {np.round(med_val,-int(np.floor(np.log10(np.abs(diff_lower)))))} or {np.round(med_val,-int(np.floor(np.log10(np.abs(diff_upper)))))}"
                        )
                        print(
                            f"Rounded Lower: {np.round(diff_lower,-int(np.floor(np.log10(np.abs(diff_lower)))))}"
                        )
                        print(
                            f"Rounded Upper: {np.round(diff_upper,-int(np.floor(np.log10(np.abs(diff_upper)))))}"
                        )
                        print("")
                    else:
                        med_val = np.median(co_bins[j])
                        diff_lower = np.median(co_bins[j]) - lower
                        diff_upper = upper - np.median(co_bins[j])
                        print(f"Median: {med_val}")
                        print(f"Lower: {lower}")
                        print(f"Upper: {upper}")
                        print(f"Diff Lower: {diff_lower}")
                        print(f"Diff Upper: {diff_upper}")
                        print(
                            f"Rounded Median: {np.round(med_val,-int(np.floor(np.log10(np.abs(diff_lower)))))} or {np.round(med_val,-int(np.floor(np.log10(np.abs(diff_upper)))))}"
                        )
                        print(
                            f"Rounded Lower: {np.round(diff_lower,-int(np.floor(np.log10(np.abs(diff_lower)))))}"
                        )
                        print(
                            f"Rounded Upper: {np.round(diff_upper,-int(np.floor(np.log10(np.abs(diff_upper)))))}"
                        )
                        print("")

    if par_sigma:
        for ax, splt_key in zip(axes, ["Mp", "M2", "COSI"]):
            if splt_key in par_sigma:
                val = par_sigma[splt_key][0]
                err = par_sigma[splt_key][1]
                fill_space_x = np.linspace(val - err, val + err, 20)
                ax.fill_between(
                    fill_space_x,
                    ax.get_ylim()[1],
                    color="grey",
                    alpha=0.2,
                    label="GLS Error",
                )
                ax.axvline(val, color="k", linestyle="--", label="GLS Best Fit Value")
            elif splt_key == "COSI" and "SINI" in par_sigma:
                sin_val, sin_err, _ = par_sigma["SINI"]
                val = np.longdouble(np.sqrt(1 - sin_val**2))
                err = np.longdouble(
                    np.sqrt((np.abs(sin_val / val)) ** 2 * sin_err**2)
                )
                fill_space_x = np.linspace(val - err, val + err, 20)
                ax.fill_between(
                    fill_space_x,
                    ax.get_ylim()[1],
                    color="grey",
                    alpha=0.2,
                    label="GLS Error",
                )
                ax.axvline(val, color="k", linestyle="--", label="GLS Best Fit Value")
            elif splt_key == "Mp":
                if "SINI" not in par_sigma and "COSI" in par_sigma:
                    cos_val, cos_err, _ = par_sigma["COSI"]
                    sin_val = np.longdouble(np.sqrt(1 - cos_val**2))
                    sin_err = np.longdouble(
                        np.sqrt((np.abs(cos_val / sin_val)) ** 2 * cos_err**2)
                    )
                else:
                    sin_val = par_sigma["SINI"][0]
                    sin_err = par_sigma["SINI"][1]

                mp = mass_pulsar(
                    par_sigma["PB"][0], par_sigma["A1"][0], sin_val, par_sigma["M2"][0]
                )

                if sin_val + sin_err > 1.0:
                    sin_up = 1.0
                else:
                    sin_up = sin_val + sin_err
                mp_err_up = mass_pulsar(
                    par_sigma["PB"][0] + par_sigma["PB"][1],
                    par_sigma["A1"][0] + par_sigma["A1"][1],
                    sin_up,
                    par_sigma["M2"][0] + par_sigma["M2"][1],
                )

                if sin_val - sin_err < 0.0:
                    sin_down = 0.0
                else:
                    sin_down = sin_val - sin_err

                if par_sigma["M2"][0] - par_sigma["M2"][1] < 0.0:
                    m2_down = 0.0
                else:
                    m2_down = par_sigma["M2"][0] - par_sigma["M2"][1]

                mp_err_down = mass_pulsar(
                    par_sigma["PB"][0] - par_sigma["PB"][1],
                    par_sigma["A1"][0] - par_sigma["A1"][1],
                    sin_down,
                    m2_down,
                )
                fill_space_x = np.linspace(mp_err_down, mp_err_up, 20)
                ax.fill_between(
                    fill_space_x,
                    ax.get_ylim()[1],
                    color="grey",
                    alpha=0.2,
                    label="GLS Errors",
                )
                ax.axvline(mp, color="k", linestyle="--", label="GLS Best Fit Value")

    # fig = plt.gcf()
    patches = []
    for jj, lab in enumerate(core_list_legend):
        patches.append(
            mpl.patches.Patch(
                color=colors[jj],
                label=lab,
                fill=False,
                linewidth=3,
                linestyle=linestyles[jj],
            )
        )  # .split(":")[-1]))

    if preliminary:
        txt = "PRELIMINARY"
        txt_loc = (0.05, 0.1)
        txt_kwargs = {"fontsize": 165, "alpha": 0.25, "rotation": 30}
        fig.text(txt_loc[0], txt_loc[1], txt, **txt_kwargs)

    if show_legend:
        fig = plt.gcf()
        allaxes = fig.get_axes()
        allaxes[0].legend(handles=patches, loc=legendloc, fontsize=legendfontsize)
    fig.subplots_adjust(wspace=wspace, hspace=hspace)

    if show_suptitle:
        plt.suptitle(
            suptitle,
            fontsize=suptitlefontsize,
            x=suptitleloc[0],
            y=suptitleloc[1],
        )
    if close:
        if show_plt:
            plt.show()
        plt.close()
