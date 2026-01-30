import numpy as np


def compute_features(raw: dict, beta_ols: float) -> dict:
    """Compute the 20 model features from ~33 raw user inputs.

    Parameters
    ----------
    raw : dict with keys like MM_NetPos_t, MM_NetPos_t1, MM_NetPos_t2,
          MM_LongPos_t/t1/t2, MM_ShortPos_t/t1/t2,
          AGG_OI_t/t1/t2, F1_OI_t, F1_OI_5d_ago, F2_OI_t, F2_OI_5d_ago,
          AGG_OI_5d_ago, F1_Price_t/t1/t2, F2_Price_t/t1/t2, F3_Price_t/t1,
          F1_Vol20D_t, F2_Vol20D_t, F3_Vol20D_t,
          Cum5D_F1_Vol_t/t1, Cum5D_F2_Vol_t/t1
    beta_ols : float — OLS beta for synthetic spread
    """
    f = {}

    # --- COT position changes (features 1-3) ---
    net_chg_t = raw["MM_NetPos_t"] - raw["MM_NetPos_t1"]
    net_chg_t1 = raw["MM_NetPos_t1"] - raw["MM_NetPos_t2"]
    f["prior_report_ManagedMoney_NetPosition_change"] = net_chg_t1

    long_chg_t = raw["MM_LongPos_t"] - raw["MM_LongPos_t1"]
    long_chg_t1 = raw["MM_LongPos_t1"] - raw["MM_LongPos_t2"]
    f["prior_report_ManagedMoney_LongPosition_change"] = long_chg_t1

    short_chg_t = raw["MM_ShortPos_t"] - raw["MM_ShortPos_t1"]
    short_chg_t1 = raw["MM_ShortPos_t1"] - raw["MM_ShortPos_t2"]
    f["prior_report_ManagedMoney_ShortPosition_change"] = short_chg_t1

    # --- Position-to-OI changes (features 4-6) ---
    net_oi_t = raw["MM_NetPos_t"] / raw["AGG_OI_t"]
    net_oi_t1 = raw["MM_NetPos_t1"] / raw["AGG_OI_t1"]
    net_oi_t2 = raw["MM_NetPos_t2"] / raw["AGG_OI_t2"]
    f["prior_report_ManagedMoney_NetPosition_to_openinterest_change"] = net_oi_t1 - net_oi_t2

    long_oi_t = raw["MM_LongPos_t"] / raw["AGG_OI_t"]
    long_oi_t1 = raw["MM_LongPos_t1"] / raw["AGG_OI_t1"]
    long_oi_t2 = raw["MM_LongPos_t2"] / raw["AGG_OI_t2"]
    f["prior_report_ManagedMoney_LongPosition_to_openinterest_change"] = long_oi_t1 - long_oi_t2

    short_oi_t = raw["MM_ShortPos_t"] / raw["AGG_OI_t"]
    short_oi_t1 = raw["MM_ShortPos_t1"] / raw["AGG_OI_t1"]
    short_oi_t2 = raw["MM_ShortPos_t2"] / raw["AGG_OI_t2"]
    f["prior_report_ManagedMoney_ShortPosition_to_openinterest_change"] = short_oi_t1 - short_oi_t2

    # --- Synthetic spread change (feature 7) ---
    spread_t = raw["F1_Price_t"] - beta_ols * raw["F2_Price_t"]
    spread_t1 = raw["F1_Price_t1"] - beta_ols * raw["F2_Price_t1"]
    spread_t2 = raw["F1_Price_t2"] - beta_ols * raw["F2_Price_t2"]
    synth_chg_t = spread_t - spread_t1
    synth_chg_t1 = spread_t1 - spread_t2
    f["prior_report_SyntheticF1MinusF2_RolledPrice_change"] = synth_chg_t1

    # --- Volume changes (features 8-10) ---
    f1_vol_chg = raw["Cum5D_F1_Vol_t"] - raw["Cum5D_F1_Vol_t1"]
    f2_vol_chg = raw["Cum5D_F2_Vol_t"] - raw["Cum5D_F2_Vol_t1"]
    f["prior_cumulative_5D_F1_Volume_change"] = f1_vol_chg
    f["prior_cumulative_5D_F2_Volume_change"] = f2_vol_chg
    f["prior_cumulative_5D_F1MinusF2_Volume_change"] = f1_vol_chg - f2_vol_chg

    # --- OI changes (features 11-14) ---
    f1_oi_chg = raw["F1_OI_t"] - raw["F1_OI_5d_ago"]
    f2_oi_chg = raw["F2_OI_t"] - raw["F2_OI_5d_ago"]
    agg_oi_chg = raw["AGG_OI_t"] - raw["AGG_OI_5d_ago"]
    f["prior_5D_F1_OI_change"] = f1_oi_chg
    f["prior_5D_F2_OI_change"] = f2_oi_chg
    f["prior_5D_AGG_OI_change"] = agg_oi_chg
    f["prior_5D_F1MinusF2_openinterest_change"] = f1_oi_chg - f2_oi_chg

    # --- Volatility (features 15-17) — user-provided directly ---
    f["F1_RolledPrice_rolling_20D_volatility"] = raw["F1_Vol20D_t"]
    f["F2_RolledPrice_rolling_20D_volatility"] = raw["F2_Vol20D_t"]
    f["F3_RolledPrice_rolling_20D_volatility"] = raw["F3_Vol20D_t"]

    # --- Price changes (features 18-20) ---
    f["F1_RolledPrice_change"] = raw["F1_Price_t"] - raw["F1_Price_t1"]
    f["F2_RolledPrice_change"] = raw["F2_Price_t"] - raw["F2_Price_t1"]
    f["F3_RolledPrice_change"] = raw["F3_Price_t"] - raw["F3_Price_t1"]

    return f
