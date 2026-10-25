# /// script
# dependencies = [
#     "marimo>=0.25.2",
#     "matplotlib==3.11.2",
#     "numpy==2.5.4",
#     "plotly==7.1.0",
# ]
# ///
"""Virtual heating appliance test bed as interactive Marimo notebook."""

import marimo

__generated_with = "0.25.2"
app = marimo.App(
    layout_file="layouts/notebook.grid.json",
    auto_download=["html"],
)


@app.cell
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import plotly.graph_objects as go

    return go, mo, np, plt


@app.cell
def _(np):
    """Virtual heating appliance test bed."""

    from typing import Final, Sequence

    H_PER_A: Final[int] = 8760

    class VirtHeiz:
        def __init__(
            self,
            q_hwb: float,
            t_hk_ausleg: tuple[float, float],
            heizkurve_steigung: float,
            a_wfl: float = 1000.0,
            t_rl_zirk=55.0,
            t_kw=15.0,
            q_twe: float = 21.0,
            q_zirk: float = 26.0,
            hydr_abgl_h: bool = False,
            factor_q_hydr_abgl_h: float = 1.1,
            hydr_abgl_twe: bool = False,
            hydr_abgl_t_adder: float = 5.0,
            twe_corr_adder: float = 10.,
            graedigkeit_wt: float = 3.0,
            t_norm: float = -16.0,
            t_hg: float = 16.0,
            h_heizstunden: float = 1800.0,
        ) -> None:
            self.q_hwb: float = q_hwb
            self.q_twe: float = q_twe
            self.q_zirk: float = q_zirk
            self.t_norm_aussen: float = t_norm
            self.heizgrenz_temp: float = t_hg
            self.heizstunden_jahr: float = h_heizstunden
            self.heizkurve_steigung: float = heizkurve_steigung
            self.graed_k: float = graedigkeit_wt
            # Adder to correct for non-idealistic potable water heating
            self.twe_corr_adder: float = twe_corr_adder

            self.vl_norm_hk: float = t_hk_ausleg[0]
            self.rl_norm_hk: float = t_hk_ausleg[1]

            self.t_rl_zirk_base: float = t_rl_zirk
            self.t_kw: float = t_kw

            self.a_wfl: float = a_wfl

            self.hydr_abgl_h: bool = hydr_abgl_h
            self.hydr_abgl_twe: bool = hydr_abgl_twe
            self.hydr_abgl_t_adder: float = hydr_abgl_t_adder

            self.faktor_heizlast: float = (
                1.0 if not hydr_abgl_h else 1.0 / factor_q_hydr_abgl_h
            )

            # Calculate everything which is not related to the current t_a:
            # Temperature spread:
            self.spreizung_norm: float = self.vl_norm_hk - self.rl_norm_hk
            # Calculate (specific) heat demand:
            self.waermebedarf_heiz: float = (
                self.a_wfl * self.q_hwb * self.faktor_heizlast
            )
            self.spez_waermebedarf_heiz: float = (
                self.q_hwb * self.faktor_heizlast
            )
            self.spez_waermebedarf_twe: float = self.q_zirk + self.q_twe
            self.spez_waermebedarf_ges: float = (
                self.spez_waermebedarf_twe + self.spez_waermebedarf_heiz
            )
            # Calculate heating power at norm conditions:
            self.norm_heizlast: float = (
                self.waermebedarf_heiz / self.heizstunden_jahr
            )
            self.p_ww: float = self.q_twe * self.a_wfl / 8760.0
            self.p_zirk: float = self.q_zirk * self.a_wfl / 8760.0
            self.p_twe_tot: float = self.p_ww + self.p_zirk

            self.t_rl_twe_wghtd: float = self._get_twe_rl()

            # Outputs:
            self.hebel_t_a: float
            self.aktuelle_heizlast: float
            self.t_vl_hk_now: float
            self.t_rl_hk_now: float
            self.t_spread_hk_now: float
            self.volumenstrom_heizung: float
            self.p_ges: float
            self.t_rl_tot: float
            self.t_rl_zirk: float
            self.lastfaktor_heizung: float
            self.lastfaktor_ges: float

        def sim_heiz(self, t_a: float) -> None:
            # Calculate current position in Heizbereich:
            self.hebel_t_a = (self.heizgrenz_temp - t_a) / (
                self.heizgrenz_temp - self.t_norm_aussen
            )
            # Calculate current heating power:
            self.aktuelle_heizlast = self.berechne_heizlast(t_a=t_a)

            self.t_vl_hk_now = self.berechne_vorlauf(t_a=t_a)
            self.t_rl_hk_now = self.berechne_ruecklauf(t_a=t_a)
            self.t_spread_hk_now = self.t_vl_hk_now - self.t_rl_hk_now

            if isinstance(t_a, float):
                self.volumenstrom_heizung = (
                    (self.aktuelle_heizlast / (1.163 * self.t_spread_hk_now))
                    if (
                        self.aktuelle_heizlast > 0.0
                        and self.t_spread_hk_now > 0.0
                    )
                    else 0.0
                )
            elif isinstance(t_a, np.ndarray | Sequence):
                self.volumenstrom_heizung = np.where(
                    np.logical_and(
                        np.array(self.aktuelle_heizlast) > 0.0,
                        np.array(self.t_spread_hk_now) > 0.0,
                    ),
                    (self.aktuelle_heizlast / (1.163 * self.t_spread_hk_now)),
                    0.0,
                )

            self.p_ges = self.aktuelle_heizlast + self.p_twe_tot
            self.lastfaktor_heizung = self.aktuelle_heizlast / self.norm_heizlast
            self.lastfaktor_ges = (self.p_ges) / (
                self.norm_heizlast + self.p_twe_tot
            )

            # Get total weighted mean rf temperature:
            self.t_rl_tot = self._get_tot_rl()

        def berechne_heizlast(self, t_a: float) -> float:
            return np.clip(
                self.norm_heizlast * self.hebel_t_a,
                a_min=0.0,
                a_max=None,
            )

        def berechne_vorlauf(self, t_a: float) -> float:
            vl_calc: float = self.vl_norm_hk - self.heizkurve_steigung * (
                t_a - self.t_norm_aussen
            )
            return np.clip(vl_calc, a_min=25.0, a_max=None)

        def berechne_ruecklauf(self, t_a: float) -> float:
            vl: float = self.berechne_vorlauf(t_a)
            base_rl: (
                float | np.typing.ndarray[[tuple[float]], np.dtype[np.float]]
            ) = np.where(
                t_a >= self.heizgrenz_temp,
                vl - 5.0,
                vl - (5.0 + (self.spreizung_norm - 5.0) * self.hebel_t_a),
            )
            # if t_a >= self.heizgrenz_temp:
            #     base_rl = vl - 5.0
            # else:
            #    self.t_spread_hk_now = 5.0 + (self.spreizung_norm - 5.0) * (
            #         self.heizgrenz_temp - t_a
            #     ) / (self.heizgrenz_temp - self.t_norm_aussen)
            #     base_rl = vl - self.t_spread_hk_now

            # Linearer Offset: +5K bei Norm-AT (-16°C), abnehmend auf 0K bei
            # Heizgrenztemp (16°C)
            zuschlag_rl_heizung: float
            if self.hydr_abgl_h:
                zuschlag_rl_heizung = 0.0
            else:
                # Linearer Interpolationsfaktor von 1.0 (bei T_norm_aussen)
                # bis 0.0 (bei heizgrenz_temp)
                faktor_offset: float = np.clip(
                    self.hebel_t_a, a_min=0.0, a_max=1.0
                )
                zuschlag_rl_heizung = self.hydr_abgl_t_adder * faktor_offset

            return base_rl + zuschlag_rl_heizung

        def _get_twe_rl(
            self,
        ) -> float:
            zuschlag_rl_zirk: float = (
                0.0 if self.hydr_abgl_twe else self.hydr_abgl_t_adder
            )
            self.t_rl_zirk = self.t_rl_zirk_base + zuschlag_rl_zirk
            return (
                self.p_ww * self.t_kw + self.p_zirk * self.t_rl_zirk
            ) / self.p_twe_tot + self.twe_corr_adder + self.graed_k

        def _get_tot_rl(
            self,
        ) -> float:
            return (
                self.aktuelle_heizlast * self.t_rl_hk_now
                + self.t_rl_twe_wghtd * self.p_twe_tot
            ) / self.p_ges + self.graed_k

    return (VirtHeiz,)


@app.cell
def _(mo):
    mo.md("""
    # Virtueller Heizungsprüfstand
    ## Heizungsrücklauftemperaturen im Teillastbetrieb einer Gebäudeheizungsanlage
    Interaktive Untersuchung der Rücklauftemperaturen eines Mehrfamilienhauses in Abhängigkeit von
    - Heizwärmebedarf, Wärmebedarf Zirkulation und WW
    - Systemtemperaturen Heizkörper, Heizkurve
    - Anlagenoptimierung: hydraulischer Abgleich
    - Außentemperatur: Teillastverhalten

    Unter den folgenden Annahmen:
    - vereinfachten Modells
    - Grädigkeit aller Wärmetauscher konstant (über alle Betriebspunkte): 3 K
    - Heizgrenztemperatur konstant: 16 °C
    - Rücklauf WW == KW mit konstant 15 °C
    """)
    return


@app.cell
def _(mo):
    wohnflaeche = mo.ui.slider(
        start=500.0,
        stop=5000.0,
        step=100.0,
        value=1000.0,
        label="Wohnfläche [m²]",
        show_value=True,
    )
    spez_heizbedarf = mo.ui.slider(
        start=20.0,
        stop=300,
        step=20.0,
        value=180,
        label="Spez. Heizwärmebedarf [kWh/(m²·a)]",
        show_value=True,
        include_input=True,
    )
    aussentemperatur = mo.ui.slider(
        start=-16,
        stop=16,
        step=2.0,
        value=-2,
        label="Außentemperatur [°C]",
        show_value=True,
    )

    zirkulationsbedarf = mo.ui.dropdown(
        options={
            "Passivhaus: 6 kWh/(m²·a)": 6.0,
            "Neubau gut: 10 kWh/(m²·a)": 10.0,
            "Neubau schlecht: 18 kWh/(m²·a)": 18.0,
            "Altbau gut: 20 kWh/(m²·a)": 20.0,
            "Altbau mittel: 26 kWh/(m²·a)": 26.0,
            "Altbau schlecht: 35 kWh/(m²·a)": 35.0,
        },
        value="Altbau mittel: 26 kWh/(m²·a)",
        label="Zirkulationswärmebedarf",
    )

    spez_ww_bedarf = mo.ui.slider(
        # start=12.5,
        # stop=30.0,
        steps=[12.5, 15.0, 19.0, 21.0, 26.0],
        value=21.0,
        label="Spez. Warmwasserbedarf [kWh/(m²·a)]",
        show_value=True,
    )

    heizkurve_steigung = mo.ui.slider(
        # start=0.2,
        # stop=2.0,
        steps=[0.4, 0.6, 0.8, 1.0, 1.4, 1.8],
        value=1.4,
        label="Steigung Heizkurve",
        show_value=True,
    )

    hk_dropdown = mo.ui.dropdown(
        options={
            "70/55 (Standard)": (70, 55),
            "90/70": (90, 70),
            "70/65": (70, 65),
            "55/45": (55, 45),
            "40/33 (Fußbodenheizung)": (40, 33),
            "35/28 (Fußbodenheizung)": (35, 28),
        },
        value="70/55 (Standard)",
        label="Systemtemp. Heizkörper<br>(VL/RL bei Norm-AT)",
    )

    hydr_abgleich_heizung = mo.ui.checkbox(
        value=False, label="hydr. Abgleich Heizung"
    )
    hydr_abgleich_twe = mo.ui.checkbox(value=False, label="hydr. Abgleich TWE")
    grad_switch = mo.ui.checkbox(value=True, label="Grädigkeit 3 K")
    twe_correction = mo.ui.slider(
        start=0.,
        stop=20.0,
        value=10.,
        label="Korrektur Delta TWE-RL",
        show_value=True,
    )
    return (
        aussentemperatur,
        grad_switch,
        heizkurve_steigung,
        hk_dropdown,
        hydr_abgleich_heizung,
        hydr_abgleich_twe,
        spez_heizbedarf,
        spez_ww_bedarf,
        wohnflaeche,
        zirkulationsbedarf,
        twe_correction
    )


@app.cell
def _(
    aussentemperatur,
    grad_switch,
    heizkurve_steigung,
    hk_dropdown,
    hydr_abgleich_heizung,
    hydr_abgleich_twe,
    mo,
    spez_heizbedarf,
    spez_ww_bedarf,
    wohnflaeche,
    zirkulationsbedarf,
    twe_correction
):
    mo.vstack(
        [
            mo.md("### Eingabeparameter"),
            mo.hstack([wohnflaeche]),
            mo.hstack([spez_heizbedarf, hk_dropdown, heizkurve_steigung]),
            mo.hstack(
                [zirkulationsbedarf, spez_ww_bedarf], justify="start", gap=2
            ),
            mo.hstack(
                [
                    hydr_abgleich_heizung,
                    hydr_abgleich_twe,
                    grad_switch,
                    twe_correction,
                ],
                justify="start",
                gap=2,
            ),
            aussentemperatur,
        ]
    )
    return


@app.cell
def _(
    VirtHeiz,
    aussentemperatur,
    grad_switch,
    heizkurve_steigung,
    hk_dropdown,
    hydr_abgleich_heizung,
    hydr_abgleich_twe,
    spez_heizbedarf,
    spez_ww_bedarf,
    wohnflaeche,
    zirkulationsbedarf,
    twe_correction,
):
    virt_heiz: VirtHeiz = VirtHeiz(
        q_hwb=spez_heizbedarf.value,
        t_hk_ausleg=hk_dropdown.value,
        heizkurve_steigung=heizkurve_steigung.value,
        a_wfl=wohnflaeche.value,
        q_twe=spez_ww_bedarf.value,
        q_zirk=zirkulationsbedarf.value,
        hydr_abgl_h=hydr_abgleich_heizung.value,
        hydr_abgl_twe=hydr_abgleich_twe.value,
        graedigkeit_wt=3.0 if grad_switch.value else 0.0,
    )
    virt_heiz_array: VirtHeiz = VirtHeiz(
        q_hwb=spez_heizbedarf.value,
        t_hk_ausleg=hk_dropdown.value,
        heizkurve_steigung=heizkurve_steigung.value,
        a_wfl=wohnflaeche.value,
        q_twe=spez_ww_bedarf.value,
        q_zirk=zirkulationsbedarf.value,
        hydr_abgl_h=hydr_abgleich_heizung.value,
        hydr_abgl_twe=hydr_abgleich_twe.value,
        graedigkeit_wt=3.0 if grad_switch.value else 0.0,
        twe_corr_adder=twe_correction.value,
    )
    virt_heiz.sim_heiz(aussentemperatur.value)
    return virt_heiz, virt_heiz_array


@app.cell
def _(
    aussentemperatur,
    go,
    heizkurve_steigung,
    hydr_abgleich_twe,
    np,
    virt_heiz: "VirtHeiz",
    virt_heiz_array: "VirtHeiz",
):
    fig = go.Figure()

    t_range = np.linspace(-16, 16, 40)
    virt_heiz_array.sim_heiz(t_range)
    # hk_vl_line = [berechne_vorlauf(t) for t in t_range]
    # hk_vl_line = [berechne_ruecklauf(t) for t in t_range]
    hk_vl_line = virt_heiz_array.t_vl_hk_now
    hk_rl_line = virt_heiz_array.t_rl_hk_now

    # faktor_h_kurve = 1.0 if hydr_abgleich_heizung.value else 1.10
    # w_bbedarf = wohnflaeche.value * spez_heizbedarf.value * faktor_h_kurve
    # p_max_val = w_bbedarf / 1800.0

    # h_last = np.array([berechne_heizlast(t_val) for t_val in t_range])
    # rl_val = np.array([berechne_ruecklauf(t_val) for t_val in t_range])
    # g_leistung = h_last + p_zirk + p_ww
    # g_rl = get_tot_rl(
    #    rl_heiz=rl_val,
    #    aktuelle_heizlast=h_last,
    #    gewichtet_rl_ww_zirk=gewichtet_rl_ww_zirk,
    #    p_ww_zirk_gesamt=p_ww_zirk_gesamt,
    #    graed_k=graed_k,
    # )
    # gewichtet_rl_line = g_rl

    gewichtet_rl_line = virt_heiz_array.t_rl_tot

    fig.add_trace(
        go.Scatter(
            x=t_range,
            y=hk_vl_line,
            mode="lines",
            name="VL Heizung",
            line=dict(color="firebrick", width=2),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=t_range,
            y=hk_rl_line,
            mode="lines",
            name="RL Heizung",
            line=dict(color="royalblue", width=2),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=t_range,
            y=gewichtet_rl_line,
            mode="lines",
            name="RL FW",
            line=dict(color="darkorchid", width=2, dash="dash"),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=[aussentemperatur.value],
            y=[virt_heiz.t_vl_hk_now],
            mode="markers+text",
            name="VL Heizung",
            text=[f"VL H: {virt_heiz.t_vl_hk_now:.1f}°C"],
            textposition="top center",
            marker=dict(color="firebrick", size=10),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[aussentemperatur.value],
            y=[virt_heiz.t_rl_hk_now],
            mode="markers+text",
            name="RL Heizung",
            text=[f"RL H: {virt_heiz.t_rl_hk_now:.1f}°C"],
            textposition="bottom center",
            marker=dict(color="royalblue", size=10),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=[aussentemperatur.value],
            y=[virt_heiz.t_rl_zirk],
            mode="markers+text",
            name="RL Zirk.",
            text=[f"RL Zirk.: {virt_heiz.t_rl_zirk:.1f}°C"],
            textposition="top right",
            marker=dict(color="peru", size=10, symbol="triangle-down"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[aussentemperatur.value],
            y=[virt_heiz.t_kw],
            mode="markers+text",
            name="RL WW",
            text=[f"RL WW: {virt_heiz.t_kw:.1f}°C"],
            textposition="bottom right",
            marker=dict(color="cadetblue", size=10, symbol="triangle-down"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[aussentemperatur.value],
            y=[virt_heiz.t_rl_twe_wghtd],
            mode="markers+text",
            name="RL TWE",
            text=[f"RL TWE: {virt_heiz.t_rl_twe_wghtd:.1f}°C"],
            textposition="bottom right",
            marker=dict(color="crimson", size=10, symbol="triangle-down"),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=[aussentemperatur.value],
            y=[virt_heiz.t_rl_tot],
            mode="markers+text",
            name=f"RL FW ({virt_heiz.t_rl_tot:.1f} °C)",
            text=[f"RL FW: {virt_heiz.t_rl_tot:.1f}°C"],
            textposition="top left",
            marker=dict(color="red", size=14, symbol="x"),
        )
    )

    fig.update_layout(
        title=f"VL- und RL-Temperaturen im MFH (Steigung = {heizkurve_steigung.value})",
        xaxis_title="T außen / °C",
        yaxis_title="T / °C",
        autosize=True,
        margin=dict(l=0, r=0, t=30, b=30, pad=4),  # Minimale Grund-Margins
        xaxis=dict(range=[-16, 16], tickmode="linear", tick0=-16, dtick=4),
        yaxis=dict(range=[10, 80]),
        # legend=dict(x=0.02, y=0.98, font=dict(size=10)),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=0.02,  # Ganz leicht über dem oberen Rand
            xanchor="left",
            x=0.02,
            font=dict(size=10),
        ),
        template="plotly_white",
        height=500,
    )

    labels = ["Heizung", "Zirk.", "WW"]
    values = [virt_heiz.aktuelle_heizlast, virt_heiz.p_zirk, virt_heiz.p_ww]

    pie_fig = go.Figure(
        data=[
            go.Pie(
                labels=labels,
                values=values,
                hole=0.4,
                textinfo="label+percent+value",
                hovertemplate="<b>%{label}</b>: %{value:.2f} kW (%{percent})<extra></extra>",
                marker=dict(colors=["royalblue", "darkorange", "teal"]),
            )
        ]
    )

    pie_fig.update_layout(
        title="Leistungsanteile im MFH bei aktueller AT",
        template="plotly_white",
        height=400,
    )

    schema_fig = go.Figure()

    schema_fig.add_trace(
        go.Scatter(
            x=[1, 1, 2.5, None, 1, 5.5, None, 5.5, 4.5, None, 5.5, 6.5],
            y=[3, 2, 2, None, 2, 2, None, 2, 1, None, 2, 1],
            mode="lines",
            line=dict(color="gray", width=3),
            showlegend=False,
        )
    )

    anteil_heizung = virt_heiz.aktuelle_heizlast / virt_heiz.p_ges
    anteil_ww_zirk = virt_heiz.p_twe_tot / virt_heiz.p_ges
    anteil_zirk = virt_heiz.p_zirk / virt_heiz.p_ges
    anteil_ww = virt_heiz.p_ww / virt_heiz.p_ges

    size_zentrale = 35
    size_heizung = max(15, min(60, 15 + anteil_heizung * 45))
    size_ww_zirk = max(15, min(60, 15 + anteil_ww_zirk * 45))
    size_zirk = max(15, min(60, 15 + anteil_zirk * 45))
    size_ww = max(15, min(60, 15 + anteil_ww * 45))

    node_x = [1, 2.5, 5.5, 4.5, 6.5]
    node_y = [3, 2, 2, 1, 1]
    node_sizes = [
        size_zentrale,
        size_heizung,
        size_ww_zirk,
        size_ww,
        size_zirk,
    ]
    node_temps = [
        virt_heiz.t_rl_tot,
        virt_heiz.t_rl_hk_now,
        virt_heiz.t_rl_twe_wghtd,
        virt_heiz.t_kw,
        virt_heiz.t_rl_zirk,
    ]

    zirk_label_extra = "" if hydr_abgleich_twe.value else ""
    dqstr: str = "Leistung"
    node_text = [
        f"FW<br>Leistung: {virt_heiz.p_ges:.1f} kW<br><b>RL: {virt_heiz.t_rl_tot:.1f} °C</b>",
        f"Heizkreise ({anteil_heizung * 100:.0f}%)<br>Leistung: {virt_heiz.aktuelle_heizlast:.1f} kW<br><b>RL: {virt_heiz.t_rl_hk_now:.1f} °C</b>",
        f"WW & Zirk. ({anteil_ww_zirk * 100:.0f}%)<br>{dqstr}: {virt_heiz.p_twe_tot:.1f} kW<br><b>RL: {virt_heiz.t_rl_twe_wghtd:.1f} °C</b>",
        f"WW ({anteil_ww * 100:.0f}%)<br>Leistung: {virt_heiz.p_ww:.1f} kW<br><b>RL: {virt_heiz.t_kw:.1f} °C</b>",
        f"Zirk.{zirk_label_extra} ({anteil_zirk * 100:.0f}%)<br>Leistung: {virt_heiz.p_zirk:.1f} kW<br><b>RL: {virt_heiz.t_rl_zirk:.1f} °C</b>",
    ]

    schema_fig.add_trace(
        go.Scatter(
            x=node_x,
            y=node_y,
            mode="markers+text",
            text=node_text,
            textposition="top center",
            marker=dict(
                size=node_sizes,
                color=node_temps,
                colorscale="Jet",
                cmin=20.0,
                cmax=70.0,
                colorbar=dict(title="T [°C]"),
                showscale=True,
                sizemode="diameter",
            ),
            showlegend=False,
        )
    )

    schema_fig.update_layout(
        title="Vereinfachtes Anlagenschema",
        autosize=True,
        margin=dict(l=0, r=0, t=30, b=30, pad=4),  # Minimale Grund-Margins
        xaxis=dict(
            showgrid=False, zeroline=False, showticklabels=False, range=[0, 8]
        ),
        yaxis=dict(
            showgrid=False, zeroline=False, showticklabels=False, range=[0, 4]
        ),
        template="plotly_white",
        height=400,
    )
    return fig, pie_fig, schema_fig, t_range


@app.cell
def _(fig, mo, schema_fig):
    mo.hstack([schema_fig, fig], gap=0.0, widths=[0.4, 0.6])
    return


@app.cell
def _(mo, pie_fig, schema_fig):
    mo.vstack([schema_fig, pie_fig])
    return


@app.cell
def _(mo, spez_ww_bedarf, virt_heiz: "VirtHeiz", zirkulationsbedarf):
    mo.md(f"""
    ### Ergebnisse & Kennwerte bei aktueller Außentemperatur

    - **Heizung**:
      - **Heizwärmebedarf:** {virt_heiz.spez_waermebedarf_heiz:.0f} kWh/(m²a)
      - **Norm-Heizlast:** {virt_heiz.norm_heizlast:.1f} kW
      - **Aktuelle Heizlast:** {virt_heiz.aktuelle_heizlast:.1f} kW
      - **Lastfaktor:** {virt_heiz.lastfaktor_heizung:.2f}
      - **VL-Temperatur:** {virt_heiz.t_vl_hk_now:.1f} °C
      - **RL-Temperatur:** {virt_heiz.t_rl_hk_now:.1f} °C
      - **Temperaturspreizung:** {virt_heiz.t_spread_hk_now:.1f} K
      - **Volumenstrom:** {virt_heiz.volumenstrom_heizung:.1f} m³/h
    - **Trinkwassererwärmung:**
      - **Wärmebedarf gesamt:** {virt_heiz.spez_waermebedarf_twe:.0f} kWh/(m²a)
      - **Wärmebedarf WW:** {spez_ww_bedarf.value:.0f} kWh/(m²a)
      - **Wärmebedarf Zirk.:** {zirkulationsbedarf.value:.0f} kWh/(m²a)
      - **Zirk.-Leistung:** {virt_heiz.p_zirk:.2f} kW (RL: {virt_heiz.t_rl_zirk:.1f} °C)
      - **WW-Leistung:** {virt_heiz.p_ww:.2f} kW (RL konst 20.0 °C)
      - **RL-Temperatur (WW + Zirk.):** {virt_heiz.t_rl_twe_wghtd:.1f} °C
    - **Gesamtsystem:**
      - **Gesamtwärmebedarf:** {virt_heiz.spez_waermebedarf_ges:.0f} kWh/(m²a)
      - **Leistungsgewichteter Mittelwert der RL-Temperaturen:** {virt_heiz.t_rl_tot:.1f} °C
      - **Lastfaktor gesamt:** {virt_heiz.lastfaktor_ges:.2f}
    """)
    return


@app.cell(disabled=True)
def _(VirtHeiz, mo, np, plt, t_range):
    # Nachrechnung verschiedener Szenarien.

    virt_systeme: dict[str, VirtHeiz] = {
        "Altbau": VirtHeiz(
            q_hwb=240.0,
            t_hk_ausleg=(90, 70),
            heizkurve_steigung=1.8,
            a_wfl=1000.0,
            q_twe=21.0,
            q_zirk=35.0,
            hydr_abgl_h=False,
            hydr_abgl_twe=False,
            graedigkeit_wt=3.0,
        ),
        "Szenario": VirtHeiz(
            q_hwb=180.0,
            t_hk_ausleg=(70, 55),
            heizkurve_steigung=1.4,
            a_wfl=1000.0,
            q_twe=21.0,
            q_zirk=26.0,
            hydr_abgl_h=False,
            hydr_abgl_twe=False,
            graedigkeit_wt=3.0,
        ),
        "Neubau - kein hydr. Abgleich": VirtHeiz(
            q_hwb=30.0,
            t_hk_ausleg=(35, 28),
            heizkurve_steigung=0.4,
            a_wfl=1000.0,
            q_twe=21.0,
            q_zirk=10.0,
            hydr_abgl_h=False,
            hydr_abgl_twe=False,
            graedigkeit_wt=3.0,
        ),
        "Neubau - optimal": VirtHeiz(
            q_hwb=30.0,
            t_hk_ausleg=(35, 28),
            heizkurve_steigung=0.4,
            a_wfl=1000.0,
            q_twe=21.0,
            q_zirk=10.0,
            hydr_abgl_h=True,
            hydr_abgl_twe=True,
            graedigkeit_wt=3.0,
        ),
    }
    virt_hwb_qz: dict[str, VirtHeiz] = {
        "Altbau, Z 35 kWh/(m²a)": VirtHeiz(
            q_hwb=240.0,
            t_hk_ausleg=(90, 70),
            heizkurve_steigung=1.8,
            a_wfl=1000.0,
            q_twe=21.0,
            q_zirk=35.0,
            hydr_abgl_h=False,
            hydr_abgl_twe=False,
            graedigkeit_wt=3.0,
        ),
        "Altbau, Z 15 kWh/(m²a)": VirtHeiz(
            q_hwb=240.0,
            t_hk_ausleg=(90, 70),
            heizkurve_steigung=1.8,
            a_wfl=1000.0,
            q_twe=21.0,
            q_zirk=15.0,
            hydr_abgl_h=False,
            hydr_abgl_twe=False,
            graedigkeit_wt=3.0,
        ),
        "Neubau, Z 18 kWh/(m²a)": VirtHeiz(
            q_hwb=30.0,
            t_hk_ausleg=(35, 28),
            heizkurve_steigung=0.4,
            a_wfl=1000.0,
            q_twe=21.0,
            q_zirk=18.0,
            hydr_abgl_h=False,
            hydr_abgl_twe=False,
            graedigkeit_wt=3.0,
        ),
        "Neubau, Z 10 kWh/(m²a)": VirtHeiz(
            q_hwb=30.0,
            t_hk_ausleg=(35, 28),
            heizkurve_steigung=0.4,
            a_wfl=1000.0,
            q_twe=21.0,
            q_zirk=10.0,
            hydr_abgl_h=True,
            hydr_abgl_twe=True,
            graedigkeit_wt=3.0,
        ),
    }
    virt_hydr_abgl: dict[str, VirtHeiz] = {
        "Altbau, kein Abgl.": VirtHeiz(
            q_hwb=240.0,
            t_hk_ausleg=(90, 70),
            heizkurve_steigung=1.8,
            a_wfl=1000.0,
            q_twe=21.0,
            q_zirk=35.0,
            hydr_abgl_h=False,
            hydr_abgl_twe=False,
            graedigkeit_wt=3.0,
        ),
        "Altbau, Abgl. H": VirtHeiz(
            q_hwb=240.0,
            t_hk_ausleg=(90, 70),
            heizkurve_steigung=1.8,
            a_wfl=1000.0,
            q_twe=21.0,
            q_zirk=35.0,
            hydr_abgl_h=True,
            hydr_abgl_twe=False,
            graedigkeit_wt=3.0,
        ),
        "Altbau, Abgl. TWE": VirtHeiz(
            q_hwb=240.0,
            t_hk_ausleg=(90, 70),
            heizkurve_steigung=1.8,
            a_wfl=1000.0,
            q_twe=21.0,
            q_zirk=35.0,
            hydr_abgl_h=False,
            hydr_abgl_twe=True,
            graedigkeit_wt=3.0,
        ),
        "Altbau, Abgl. H+TWE": VirtHeiz(
            q_hwb=240.0,
            t_hk_ausleg=(90, 70),
            heizkurve_steigung=1.8,
            a_wfl=1000.0,
            q_twe=21.0,
            q_zirk=35.0,
            hydr_abgl_h=True,
            hydr_abgl_twe=True,
            graedigkeit_wt=3.0,
        ),
        "Neubau, kein Abgl.": VirtHeiz(
            q_hwb=30.0,
            t_hk_ausleg=(35, 28),
            heizkurve_steigung=0.4,
            a_wfl=1000.0,
            q_twe=21.0,
            q_zirk=10.0,
            hydr_abgl_h=False,
            hydr_abgl_twe=False,
            graedigkeit_wt=3.0,
        ),
        "Neubau, Abgl. H": VirtHeiz(
            q_hwb=30.0,
            t_hk_ausleg=(35, 28),
            heizkurve_steigung=0.4,
            a_wfl=1000.0,
            q_twe=21.0,
            q_zirk=10.0,
            hydr_abgl_h=True,
            hydr_abgl_twe=False,
            graedigkeit_wt=3.0,
        ),
        "Neubau, Abgl. TWE": VirtHeiz(
            q_hwb=30.0,
            t_hk_ausleg=(35, 28),
            heizkurve_steigung=0.4,
            a_wfl=1000.0,
            q_twe=21.0,
            q_zirk=10.0,
            hydr_abgl_h=False,
            hydr_abgl_twe=True,
            graedigkeit_wt=3.0,
        ),
        "Neubau, Abgl. H+TWE": VirtHeiz(
            q_hwb=30.0,
            t_hk_ausleg=(35, 28),
            heizkurve_steigung=0.4,
            a_wfl=1000.0,
            q_twe=21.0,
            q_zirk=10.0,
            hydr_abgl_h=True,
            hydr_abgl_twe=True,
            graedigkeit_wt=3.0,
        ),
    }
    virt_qz_vs_qtwe_wfl: dict[str, VirtHeiz] = {
        "EnEV 02, Z 35 kWh/(m²a)": VirtHeiz(
            q_hwb=80.0,
            t_hk_ausleg=(40, 33),
            heizkurve_steigung=0.6,
            a_wfl=1000.0,
            q_twe=21.0,
            q_zirk=35.0,
            hydr_abgl_h=False,
            hydr_abgl_twe=False,
            graedigkeit_wt=3.0,
        ),
        "EnEV 02, Z 10 kWh/(m²a)": VirtHeiz(
            q_hwb=80.0,
            t_hk_ausleg=(40, 33),
            heizkurve_steigung=0.6,
            a_wfl=1000.0,
            q_twe=21.0,
            q_zirk=10.0,
            hydr_abgl_h=False,
            hydr_abgl_twe=False,
            graedigkeit_wt=3.0,
        ),
        "Nebau, Wfl 1000m²": VirtHeiz(
            q_hwb=30.0,
            t_hk_ausleg=(35, 28),
            heizkurve_steigung=0.4,
            a_wfl=1000.0,
            q_twe=21.0,
            q_zirk=10.0,
            hydr_abgl_h=False,
            hydr_abgl_twe=False,
            graedigkeit_wt=3.0,
        ),
        "Nebau, Wfl 5000m²": VirtHeiz(
            q_hwb=30.0,
            t_hk_ausleg=(35, 28),
            heizkurve_steigung=0.4,
            a_wfl=1000.0,
            q_twe=21.0,
            q_zirk=10.0,
            hydr_abgl_h=False,
            hydr_abgl_twe=False,
            graedigkeit_wt=3.0,
        ),
    }
    colors: list[str] = ["C3", "C1", "C0", "C2", "C3", "C1", "C0", "C2"]
    linestyles: list[str] = ["-", "-", "-", "-", "-.", "-.", "-.", "-."]

    fig_b, axs = plt.subplots(2, 2, sharex=True, sharey=True)
    width_cm: float = 30.0
    fig_b.set_size_inches(width_cm / 2.54, width_cm / 2.54 * 9 / 16)
    # Szenarien
    axs[0, 0].set_title("Einfluss Gebäudezustand")
    for i, (k, v) in enumerate(virt_systeme.items()):
        v.sim_heiz(t_range)

        axs[0, 0].plot(
            t_range, v.t_rl_tot, label=k, color=colors[i], ls=linestyles[i]
        )

    # Einfluss Zirk vs. HWB
    axs[0, 1].set_title("Einfluss Zirkulation vs. Heizwärmebedarf")
    for i, (k, v) in enumerate(virt_hwb_qz.items()):
        v.sim_heiz(t_range)

        axs[0, 1].plot(
            t_range, v.t_rl_tot, label=k, color=colors[i], ls=linestyles[i]
        )

    # Einfluss hydr. Abgleich vs. Baujahr
    axs[1, 0].set_title("Einfluss hydraulischer Abgleich vs. Baujahr")
    for i, (k, v) in enumerate(virt_hydr_abgl.items()):
        v.sim_heiz(t_range)

        axs[1, 0].plot(
            t_range, v.t_rl_tot, label=k, color=colors[i], ls=linestyles[i]
        )

    # Einfluss hydr. Abgleich vs. Baujahr
    axs[1, 1].set_title("Einfluss Wohnfläche sowie Wärmebedarf Zirk. vs. WW")
    linestyles_wfl = ["-", "-", "-.", ":"]
    for i, (k, v) in enumerate(virt_qz_vs_qtwe_wfl.items()):
        v.sim_heiz(t_range)

        if i < 2:
            axs[1, 1].plot(
                t_range,
                np.repeat(v.t_rl_twe_wghtd, t_range.size),
                label=k,
                color=colors[i],
                ls=linestyles_wfl[i],
            )
        else:
            axs[1, 1].plot(
                t_range,
                v.t_rl_tot,
                label=k,
                color=colors[i],
                ls=linestyles_wfl[i],
            )
    axs[1, 1].set_ylabel("$T_{RL,TWE}$ / °C")
    axs[1, 1].yaxis.set_label_position("right")

    axs[0, 0].set_ylabel("$T_{RL,FW}$ / °C")
    axs[1, 0].set_ylabel("$T_{RL,FW}$ / °C")
    axs[1, 0].set_xlabel("$T_{a}$ / °C")
    axs[1, 1].set_xlabel("$T_{a}$ / °C")
    axs[0, 0].set_ylim(20, 80)
    axs[0, 0].margins(0.0, 0.0)
    for ax in axs.flat:
        ax.legend()
        ax.grid(True)

    axs[1, 0].legend(ncols=2)

    plt.tight_layout()
    plt.savefig("Einfluss_auf_RL.png", dpi=300)

    axmo = mo.ui.matplotlib(plt.gca())
    axmo
    return colors, linestyles


@app.cell(disabled=True)
def _(VirtHeiz, colors: list[str], linestyles: list[str], mo, plt, t_range):
    # Abgleich neu-alt.

    virt_neu_vs_alt: dict[str, VirtHeiz] = {
        "vor Maßnahmen": VirtHeiz(
            q_hwb=180.0,
            t_hk_ausleg=(70, 55),
            heizkurve_steigung=1.4,
            a_wfl=1000.0,
            q_twe=21.0,
            q_zirk=26.0,
            hydr_abgl_h=False,
            hydr_abgl_twe=False,
            graedigkeit_wt=3.0,
        ),
        "nach Maßnahmen": VirtHeiz(
            q_hwb=50.0,
            t_hk_ausleg=(55, 45),
            heizkurve_steigung=1.0,
            a_wfl=1000.0,
            q_twe=21.0,
            q_zirk=16.0,
            hydr_abgl_h=True,
            hydr_abgl_twe=True,
            graedigkeit_wt=3.0,
        ),
    }

    colors_c: list[str] = ["C3", "C1", "C0", "C2", "C3", "C1", "C0", "C2"]
    linestyles_c: list[str] = ["-", "-", "-", "-", "-.", "-.", "-.", "-."]

    fig_c, ax_c = plt.subplots(1, 1, sharex=True, sharey=True)
    width_cm_c: float = 20.0
    fig_c.set_size_inches(width_cm_c / 2.54, width_cm_c / 2.54 * 9 / 16)
    # Szenarien
    ax_c.set_title(
        "Szenario: Initiale Ausgangslage vs. Umsetzung aller Maßnahmen"
    )
    for j, (kk, vv) in enumerate(virt_neu_vs_alt.items()):
        vv.sim_heiz(t_range)
        ax_c.plot(
            t_range, vv.t_rl_tot, label=kk, color=colors[j], ls=linestyles[j]
        )

    ax_c.fill_between(
        t_range,
        virt_neu_vs_alt["vor Maßnahmen"].t_rl_tot,
        virt_neu_vs_alt["nach Maßnahmen"].t_rl_tot,
        color="palegreen",
        label="Reduktion",
    )

    ax_c.set_ylabel("$T_{RL,FW}$ / °C")
    ax_c.set_xlabel("$T_{a}$ / °C")
    ax_c.set_ylim(20, 80)
    ax_c.margins(0.0, 0.0)
    ax_c.legend()
    ax_c.grid(True)

    ax_c.legend(ncols=1)

    plt.tight_layout()
    plt.savefig("Szenario-alt-vs-neu.png", dpi=300)

    axmo_c = mo.ui.matplotlib(plt.gca())
    axmo_c
    return


if __name__ == "__main__":
    app.run()
