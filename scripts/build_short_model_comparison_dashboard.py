"""Create a standalone dashboard comparing the three short-model portfolios.

All scenarios apply the gap rule before ranking and filling the three daily
slots.  The 4% stop / 8% target scenario uses the conservative stop-first
rule when a daily candle touches both levels.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from plotly.offline import get_plotlyjs

from analyze_priority_short_take_profit import (
    FEES,
    candidates,
    load_fee_schedule,
    performance_summary,
    simulate,
    tabular_statistics,
)


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "artifacts/short_model_gap_down3_comparison_dashboard"
MAX_NEGATIVE_GAP = -0.03
MAX_POSITIONS = 3
ALLOCATIONS = (0.20, 0.33)
EXIT_RULES = (
    ("SL 5% / close exit", 0.05, None, "close_only"),
    ("SL 10% / close exit", 0.10, None, "close_only"),
    ("SL 4% / TP 8% (conservative)", 0.04, 0.08, "stop_first"),
)
MODELS = (
    ("baseline", "Baseline q99.97"),
    ("raw_tail", "Corrected raw-tail q99.97"),
    ("merged", "Priority merge: Baseline then raw-tail"),
)


def select_for_model(frame: pd.DataFrame, model: str) -> pd.DataFrame:
    """Gap filter first, then select the chosen model portfolio's three slots."""
    eligible = frame.loc[frame["preselected"] & frame["signal_day_turnover"].ge(5_000_000)].copy()
    eligible["opening_gap_return"] = eligible["target_open"].div(eligible["signal_close"]).sub(1.0)
    eligible = eligible.loc[eligible["opening_gap_return"].gt(MAX_NEGATIVE_GAP)].copy()
    if model == "baseline":
        eligible = eligible.loc[eligible["signal_source"].eq("baseline_q9997")]
        sort_columns = ["signal_date", "source_score", "symbol"]
        ascending = [True, False, True]
    elif model == "raw_tail":
        eligible = eligible.loc[eligible["signal_source"].eq("raw_tail_q9997")]
        sort_columns = ["signal_date", "source_score", "symbol"]
        ascending = [True, False, True]
    elif model == "merged":
        sort_columns = ["signal_date", "signal_source_priority", "source_score", "symbol"]
        ascending = [True, True, False, True]
    else:
        raise ValueError(model)
    return (
        eligible.sort_values(sort_columns, ascending=ascending, kind="stable")
        .drop_duplicates(["signal_date", "symbol"], keep="first")
        .groupby("signal_date", sort=False)
        .head(MAX_POSITIONS)
        .copy()
    )


def safe(value: float | int | np.number | None, digits: int = 8) -> float | int | None:
    if value is None or not np.isfinite(value):
        return None
    if isinstance(value, (int, np.integer)):
        return int(value)
    return round(float(value), digits)


def summary_payload(
    summary: dict[str, object],
    stats: dict[str, object],
    selected: pd.DataFrame,
    ambiguity: dict[str, int],
) -> dict[str, float | int | None]:
    return {
        "Total return (%)": safe(100 * float(summary["total_return"]), 2),
        "Sum gross trade returns (%)": safe(100 * float(stats["sum_gross_return"]), 2),
        "Sum net trade returns (%)": safe(100 * float(stats["sum_net_return"]), 2),
        "CAGR (%)": safe(100 * float(summary["cagr"]), 2),
        "Max drawdown (%)": safe(100 * float(summary["max_drawdown"]), 2),
        "Sharpe": safe(float(summary["sharpe"]), 3),
        "Trades": int(stats["total_trades"]),
        "Positive trades": int(stats["positive_trades"]),
        "Negative trades": int(stats["negative_trades"]),
        "Win rate (%)": safe(100 * float(stats["win_rate"]), 2),
        "Trade profit factor": safe(float(stats["profit_factor"]), 3),
        "Avg. net/trade (%)": safe(100 * float(stats["average_net_return_per_trade"]), 3),
        "Target / stop / close exits": f"{stats['profit_target_exits']} / {stats['hard_stop_exits']} / {stats['close_exits']}",
        "Same-day TP/SL collisions": int(ambiguity["both_hits"]),
        "TP/SL collision rate (%)": safe(100 * ambiguity["both_hits"] / max(stats["total_trades"], 1), 2),
        "Selected signal days": int(selected["signal_date"].nunique()),
    }


def yearly_payload(trades: pd.DataFrame, daily: pd.DataFrame) -> list[dict[str, float | int | None]]:
    trade_year = trades.assign(year=pd.to_datetime(trades["target_date"]).dt.year)
    daily_year = daily.assign(year=pd.to_datetime(daily["target_date"]).dt.year)
    result: list[dict[str, float | int | None]] = []
    for year in sorted(daily_year["year"].unique()):
        stats = tabular_statistics(
            trade_year.loc[trade_year["year"].eq(year)],
            daily_year.loc[daily_year["year"].eq(year)],
        )
        result.append({
            "year": int(year),
            "trades": int(stats["total_trades"]),
            "positive": int(stats["positive_trades"]),
            "negative": int(stats["negative_trades"]),
            "win_rate_pct": safe(100 * float(stats["win_rate"]), 2),
            "profit_factor": safe(float(stats["profit_factor"]), 3),
            "avg_net_trade_pct": safe(100 * float(stats["average_net_return_per_trade"]), 3),
            "return_pct": safe(100 * float(stats["portfolio_return"]), 2),
            "max_drawdown_pct": safe(100 * float(stats["max_drawdown"]), 2),
        })
    return result


def scenario_key(model: str, allocation: float, rule: str) -> str:
    return f"{model}__alloc{int(allocation * 100)}__{rule.replace(' ', '_').replace('/', '_')}"


def dashboard_html(payload: dict[str, object]) -> str:
    plotly_js = get_plotlyjs()
    data = json.dumps(payload, separators=(",", ":"))
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Short model comparison dashboard</title>
<style>
body{{margin:0;background:#f4f7fb;color:#172033;font-family:Arial,sans-serif}} main{{max-width:1520px;margin:auto;padding:22px}}
h1{{margin:0 0 6px;font-size:27px}} .note{{margin:0 0 16px;color:#46556e;line-height:1.45}} .controls,.card,#heatmap,#equity,#years{{background:#fff;border-radius:10px;box-shadow:0 1px 4px #17203318}}
.controls{{display:grid;grid-template-columns:repeat(6,minmax(150px,1fr));gap:12px;padding:15px}} label{{display:grid;gap:5px;font-size:13px;font-weight:600}} select{{padding:8px;border:1px solid #cbd5e1;border-radius:6px;background:#fff}}
#metrics{{display:grid;grid-template-columns:1fr 1fr;gap:12px;margin:15px 0}} .card{{padding:14px}} .card h2{{margin:0 0 9px;font-size:16px}} .metricgrid{{display:grid;grid-template-columns:repeat(4,1fr);gap:7px;font-size:12px}} .metricgrid div{{padding:7px;background:#f7f9fc;border-radius:5px}} .metricgrid b{{display:block;margin-top:3px;font-size:14px}}
.chart{{margin-top:15px;min-height:500px}} #years{{padding:15px;margin-top:15px;overflow:auto}} table{{border-collapse:collapse;width:100%;font-size:13px}} th,td{{padding:7px;border-bottom:1px solid #e5eaf1;text-align:right}} th:first-child,td:first-child{{text-align:left}} th{{background:#f7f9fc}} .legend{{margin-top:8px;font-size:12px;color:#596780}}
@media(max-width:900px){{.controls{{grid-template-columns:1fr 1fr}}#metrics{{grid-template-columns:1fr}}.metricgrid{{grid-template-columns:repeat(2,1fr)}}}}
</style><script>{plotly_js}</script></head><body><main>
<h1>Short-model comparison dashboard</h1>
<p class="note">Compare Baseline q99.97, Corrected raw-tail, and their priority merge. Every configuration rejects gap-down candidates at or below 3% <strong>before</strong> model ranking and the three-slot cap. The 4% / 8% target uses the conservative stop-first assumption. “Sum net trade returns” is the arithmetic sum of each trade's after-cost return on its own position budget; “Total return” is the compounded portfolio result.</p>
<section class="controls"><label>Configuration A<select id="a"></select></label><label>Configuration B<select id="b"></select></label><label>Heatmap view<select id="view"><option value="a">Configuration A</option><option value="b">Configuration B</option><option value="diff">A minus B</option></select></label><label>Heatmap level<select id="level"><option value="month">Monthly</option><option value="year">Yearly</option><option value="day">Daily calendar</option></select></label><label id="yearbox">Calendar year<select id="year"></select></label><label>Values<select id="labels"><option value="on">Show values</option><option value="off">Hide values</option></select></label></section>
<section id="metrics"></section><div id="equity" class="chart"></div><div id="heatmap" class="chart"></div><section id="years"></section>
<script>
const DATA={data}, keys=Object.keys(DATA), el=id=>document.getElementById(id), MONTHS=["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"];
for(const id of ["a","b"]) for(const k of keys){{const o=document.createElement("option");o.value=k;o.textContent=DATA[k].label;el(id).appendChild(o)}}
el("a").value=keys.find(k=>k.startsWith("baseline__alloc20")&&k.includes("TP_8"))||keys[0];el("b").value=keys.find(k=>k.startsWith("merged__alloc20")&&k.includes("TP_8"))||keys[keys.length-1];
const YEARS=[...new Set(DATA[keys[0]].daily.map(x=>x.d.slice(0,4)))];for(const y of YEARS){{const o=document.createElement("option");o.value=y;o.textContent=y;el("year").appendChild(o)}}el("year").value=YEARS[YEARS.length-1];
const pct=v=>v===null||v===undefined?"-":`${{v.toFixed(2)}}%`;const num=v=>v===null||v===undefined?"-":String(v);
function card(k,tag){{const m=DATA[k].metrics;return `<article class="card"><h2>${{tag}}: ${{DATA[k].label}}</h2><div class="metricgrid">${{Object.entries(m).map(([n,v])=>`<div>${{n}}<b>${{num(v)}}</b></div>`).join("")}}</div></article>`}}
function rec(k){{return DATA[k].daily.map(x=>{{const z=new Date(x.d+"T00:00:00");return {{...x,y:z.getFullYear(),m:z.getMonth(),day:z.getDate()}}}})}}
function compounded(v){{return (v.reduce((a,x)=>a*(1+x/100),1)-1)*100}} function matrix(k,level,year){{const r=rec(k);if(level==="day"){{let z=Array.from({{length:31}},()=>Array(12).fill(null));for(const x of r)if(x.y===+year)z[x.day-1][x.m]=x.r;return z}}if(level==="month")return YEARS.map(y=>MONTHS.map((_,m)=>compounded(r.filter(x=>x.y===+y&&x.m===m).map(x=>x.r))));return [YEARS.map(y=>compounded(r.filter(x=>x.y===+y).map(x=>x.r)))];}}
function minus(a,b){{return a.map((r,i)=>r.map((v,j)=>v===null||b[i][j]===null?null:v-b[i][j]))}} function cap(z){{const q=z.flat().filter(Number.isFinite).map(Math.abs);return Math.max(.01,...q)}}
function equity(k){{let value=100;return DATA[k].daily.map(x=>{{value*=1+x.r/100;return {{x:x.d,y:value}}}})}}
function yearRows(k){{return Object.fromEntries(DATA[k].years.map(x=>[x.year,x]))}} function table(a,b){{const A=yearRows(a),B=yearRows(b);const rows=YEARS.map(y=>{{const x=A[y]||{{}},z=B[y]||{{}};return `<tr><td>${{y}}</td><td>${{x.trades??0}}</td><td>${{pct(x.return_pct)}}</td><td>${{pct(x.max_drawdown_pct)}}</td><td>${{num(x.profit_factor)}}</td><td>${{x.trades?pct(x.win_rate_pct):"-"}}</td><td>${{z.trades??0}}</td><td>${{pct(z.return_pct)}}</td><td>${{pct(z.max_drawdown_pct)}}</td><td>${{num(z.profit_factor)}}</td><td>${{z.trades?pct(z.win_rate_pct):"-"}}</td></tr>`}}).join("");el("years").innerHTML=`<h2>Year-wise comparison</h2><table><thead><tr><th>Year</th><th colspan="5">A: ${{DATA[a].short_label}}</th><th colspan="5">B: ${{DATA[b].short_label}}</th></tr><tr><th></th><th>Trades</th><th>Return</th><th>Max DD</th><th>PF</th><th>Win</th><th>Trades</th><th>Return</th><th>Max DD</th><th>PF</th><th>Win</th></tr></thead><tbody>${{rows}}</tbody></table><p class="legend">PF is net trade-level gross profit divided by gross loss. Returns are compounded portfolio returns for that calendar year.</p>`}}
function draw(){{const a=el("a").value,b=el("b").value,view=el("view").value,level=el("level").value,year=el("year").value;el("metrics").innerHTML=card(a,"A")+card(b,"B");el("yearbox").style.visibility=level==="day"?"visible":"hidden";const eq=[{{x:equity(a).map(p=>p.x),y:equity(a).map(p=>p.y),name:"A: "+DATA[a].short_label,type:"scatter",mode:"lines",line:{{width:2}}}},{{x:equity(b).map(p=>p.x),y:equity(b).map(p=>p.y),name:"B: "+DATA[b].short_label,type:"scatter",mode:"lines",line:{{width:2,dash:"dot"}}}}];Plotly.react("equity",eq,{{title:"Equity growth (starting capital = 100)",yaxis:{{title:"Portfolio value"}},margin:{{l:65,r:25,t:55,b:45}},height:430,legend:{{orientation:"h"}}}},{{responsive:true,displaylogo:false}});let A=matrix(a,level,year),B=matrix(b,level,year),z=view==="a"?A:view==="b"?B:minus(A,B),x=level==="year"?YEARS:MONTHS,y=level==="day"?Array.from({{length:31}},(_,i)=>i+1):level==="month"?YEARS:["Return"],title=level==="day"?`Daily return - ${{year}}`:level==="month"?"Compounded monthly return":"Compounded yearly return";if(view==="diff")title+=" (A minus B)";const d=level==="day"?2:1,trace={{type:"heatmap",z,x,y,zmid:0,zmin:-cap(z),zmax:cap(z),colorscale:"RdYlGn",text:z.map(r=>r.map(v=>Number.isFinite(v)?v.toFixed(d):"")),texttemplate:el("labels").value==="on"?"%{{text}}":"",textfont:{{size:level==="day"?9:13,color:"#172033"}},hovertemplate:"%{{y}} / %{{x}}<br>%{{z:.3f}}%<extra></extra>",colorbar:{{title:"Return (%)"}}}};Plotly.react("heatmap",[trace],{{title,margin:{{l:70,r:85,t:55,b:55}},height:level==="day"?760:510,yaxis:{{autorange:level==="day"?"reversed":true}}}},{{responsive:true,displaylogo:false}});table(a,b)}}
for(const id of ["a","b","view","level","year","labels"])el(id).addEventListener("change",draw);draw();
</script></main></body></html>"""


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    frame = candidates()
    signal_dates = pd.DatetimeIndex(frame["signal_date"].unique()).sort_values()
    fee_periods = load_fee_schedule(FEES)
    configurations: dict[str, dict[str, object]] = {}
    summary_rows: list[dict[str, object]] = []
    yearly_rows: list[dict[str, object]] = []
    selection_rows: list[pd.DataFrame] = []
    trade_rows: list[pd.DataFrame] = []
    daily_rows: list[pd.DataFrame] = []
    for model, model_label in MODELS:
        selected = select_for_model(frame, model)
        selection_rows.append(selected.assign(model_portfolio=model_label))
        for allocation in ALLOCATIONS:
            for rule_label, stop, target, policy in EXIT_RULES:
                trades, daily, ambiguity = simulate(
                    signal_dates, selected, fee_periods,
                    hard_stop_loss_pct=stop,
                    take_profit_pct=target,
                    both_hit_policy="stop_first" if policy == "close_only" else policy,
                    allocation_per_position=allocation,
                )
                summary = performance_summary(trades, daily, 100_000)
                stats = tabular_statistics(trades, daily)
                key = scenario_key(model, allocation, rule_label)
                label = f"{model_label} | {int(allocation * 100)}% each | {rule_label}"
                years = yearly_payload(trades, daily)
                configurations[key] = {
                    "label": label,
                    "short_label": f"{model_label}; {int(allocation * 100)}%; {rule_label}",
                    "metrics": summary_payload(summary, stats, selected, ambiguity),
                    "daily": [{"d": date.strftime("%Y-%m-%d"), "r": safe(100 * value, 10)} for date, value in zip(daily["target_date"], daily["portfolio_return"], strict=True)],
                    "years": years,
                }
                summary_rows.append({"key": key, "model_portfolio": model_label, "allocation_per_position": allocation, "exit_rule": rule_label, **summary_payload(summary, stats, selected, ambiguity)})
                yearly_rows.extend({"key": key, "model_portfolio": model_label, "allocation_per_position": allocation, "exit_rule": rule_label, **row} for row in years)
                trade_rows.append(trades.assign(key=key, model_portfolio=model_label, allocation_per_position=allocation, exit_rule=rule_label))
                daily_rows.append(daily.assign(key=key, model_portfolio=model_label, allocation_per_position=allocation, exit_rule=rule_label))
    pd.DataFrame(summary_rows).to_csv(OUTPUT / "comparison_summary.csv", index=False)
    pd.DataFrame(yearly_rows).to_csv(OUTPUT / "comparison_yearwise.csv", index=False)
    pd.concat(selection_rows, ignore_index=True).to_parquet(OUTPUT / "selected_signals_by_model.parquet", index=False)
    pd.concat(trade_rows, ignore_index=True).to_parquet(OUTPUT / "comparison_trade_ledger.parquet", index=False)
    pd.concat(daily_rows, ignore_index=True).to_parquet(OUTPUT / "comparison_daily_equity.parquet", index=False)
    payload = {"configurations": configurations}
    # Keep the browser payload flat; this is easier for the standalone JavaScript.
    (OUTPUT / "dashboard_data.json").write_text(json.dumps(configurations, separators=(",", ":")), encoding="utf-8")
    html = dashboard_html(configurations)
    (OUTPUT / "short_model_comparison_dashboard.html").write_text(html, encoding="utf-8")
    print(f"Wrote {OUTPUT / 'short_model_comparison_dashboard.html'}")


if __name__ == "__main__":
    main()
