"""Virtual heating appliance test bed."""

from typing import Final, Sequence

import numpy as np

H_PER_A: Final[int] = 8760


class VirtHeiz:
    """A virtual building energy system testbed for return flow assessment."""
    
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
        self.spez_waermebedarf_heiz: float = self.q_hwb * self.faktor_heizlast
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
        self.teillastfaktor: float

    def sim_heiz(
        self, 
        t_a: float | np.ndarray[tuple[int], np.dtype[np.float64]]
    ) -> None:
        """
        Simulate the heating system.

        Parameters
        ----------
        t_a
            Ambient temperature in degree celsius [°C], either as
            a single float or an array of float.
        """
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
                    self.aktuelle_heizlast > 0.0 and self.t_spread_hk_now > 0.0
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
        self.teillastfaktor = self.aktuelle_heizlast / self.norm_heizlast

        self.t_rl_tot = self._get_tot_rl()

    def berechne_heizlast(self, t_a: float) -> float:
        """
        Berechne die Heizlast.

        Parameters
        ----------
        t_a
            Ambient temperature in degree celsius [°C], either as
            a single float or an array of float.
        """
        return np.clip(
            self.norm_heizlast * self.hebel_t_a,
            a_min=0.0,
            a_max=None,
        )

    def berechne_vorlauf(self, t_a: float) -> float:
        """
        Berechne die Vorlauftemperatur.

        Parameters
        ----------
        t_a
            Ambient temperature in degree celsius [°C], either as
            a single float or an array of float.
        """
        vl_calc: float = self.vl_norm_hk - self.heizkurve_steigung * (
            t_a - self.t_norm_aussen
        )
        return np.clip(vl_calc, a_min=25.0, a_max=None)

    def berechne_ruecklauf(self, t_a: float) -> float:
        """
        Berechne die Rücklauftemperatur.

        Parameters
        ----------
        t_a
            Ambient temperature in degree celsius [°C], either as
            a single float or an array of float.
        """
        vl: float = self.berechne_vorlauf(t_a)
        base_rl: (
            float | np.typing.ndarray[[tuple[float]], np.dtype[np.float]]
        ) = np.where(
            t_a >= self.heizgrenz_temp,
            vl - 5.0,
            vl - (5.0 + (self.spreizung_norm - 5.0) * self.hebel_t_a),
        )

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
        ) / self.p_twe_tot + self.graed_k

    def _get_tot_rl(
        self,
    ) -> float:
        return (
            self.aktuelle_heizlast * self.t_rl_hk_now
            + self.t_rl_twe_wghtd * self.p_twe_tot
        ) / self.p_ges + self.graed_k
