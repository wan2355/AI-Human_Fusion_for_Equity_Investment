# How to Read the Data

The CSV files and charts generated in this repository are **not designed to rank stocks or make buy/sell decisions**.

Their purpose is to **visualize structural differences between companies** and to help the user understand
*what kind of business each company actually is*.

High or low numbers should not be interpreted in isolation.
Context and structure always matter.

---

## 1. How to Read Score_CYCLE

Score_CYCLE represents how well a company’s business structure aligns with:

- economic cycles
- capital expenditure cycles
- demand fluctuations

In short, it measures **cyclical adaptability**, not investment quality.

### Important Notes
- A higher score does NOT mean “better investment”
- It does NOT indicate safety, valuation, or upside

### Correct Usage
Use Score_CYCLE to ask:
> “In the current macro or industry phase,  
> is this company structurally suited to perform well?”

---

## 2. How to Read PEG-like  
((MC/Rev) / RevGrowth%)

The PEG-like metric is a **consistency check** between:

- market valuation (MC/Rev)
- revenue growth rate (RevGrowth%)

### Why values can become extreme
- When growth slows temporarily, the denominator shrinks
- This causes the ratio to spike, especially for expectation-driven stocks

### Common Misinterpretations
- It does NOT mean efficiency
- It does NOT mean attractiveness
- It does NOT mean over/undervaluation by itself

### Recommended Usage
- Always inspect MC/Rev and RevGrowth separately
- Use PEG-like only as a *warning signal* for valuation-growth mismatch

---

## 3. How to Read MC/FCF (P/FCF-type metrics)

MC/FCF shows how much the market is paying for current free cash flow.

### Important Caveats
- A high value does NOT automatically mean “overvalued”
- Growth companies often suppress FCF intentionally via:
  - reinvestment
  - SBC
  - capital expenditures

### Proper Interpretation
- Always pair with FCF/Rev
- Ask whether FCF is:
  - realized today, or
  - intentionally deferred for future growth

---

## 4. How to Read the Three Axes: CYCLE / SW / HW

The axes shown in `summary.csv` are **not scores**.
They describe **business terrain**, not performance ranking.

### CYCLE
- Sensitivity to macro cycles and investment waves
- Timing-dependent business structure

### SW (Software-like)
- Human-capital driven
- High scalability and margin leverage
- Time tends to work in favor of this structure

### HW (Hardware-like)
- Asset- and CapEx-intensive
- High barriers to entry
- Capital efficiency depends on cycle timing

### Critical Warning
- Do NOT sum these values
- Do NOT rank companies by total score

### Correct Perspective
Use these axes to consider:
- Which business structures fit the current environment
- Whether portfolio exposure is structurally diversified

---

## Final Notes

These metrics are **thinking tools**, not decision engines.

They are meant to:
- reduce misunderstanding
- highlight structural differences
- support deliberate reasoning

They should never replace judgment.

