# Beaver's Choice Paper Company: Multi-Agent System Reflection Report

**Framework:** pydantic-ai · **Model:** gpt-4o-mini (Vocareum OpenAI proxy) · **Agents:** 5 (1 orchestrator + 4 workers)

**Deliverables:** `workflow_diagram.png`, `project_starter.py` (the single source file), `test_results.csv`, and this report. Supporting files are `business_advisor_report.md` and `evaluation_run_log.txt`, which is the full trace of the evaluation run.

---

## 1. The agent workflow and why it is built this way

### 1.1 Agents and responsibilities

The system has five agents, the maximum allowed. Each one owns a distinct part of the business, so no two agents write the same kind of record or make the same decision.

| Agent | Owns | Decides | Writes to the ledger |
|---|---|---|---|
| **1. Orchestrator** | The conversation with the customer | Which items were requested, the deadline, whether the customer wants to buy or only wants a quote, and what to tell the customer | Nothing |
| **2. Inventory Agent** | Stock and purchasing | Whether each line is in stock, back-ordered, past the deadline, not carried, or over budget; when to restock | `stock_orders` (back-orders and replenishment) |
| **3. Quoting Agent** | Pricing | How to present the price and the discount, using similar past quotes | Nothing |
| **4. Sales Agent** | Closing the sale | Whether the sale is financially safe to book (cash reserve, selling above cost) | `sales` |
| **5. Business Advisor** | Management insight | Which operational changes would improve revenue and efficiency | Nothing (it reports) |

### 1.2 Orchestration and data flow (numbers match the diagram)

1. The customer's request, stamped with its date, goes to the **Orchestrator**. The Orchestrator calls `record_customer_order`, which extracts every item with its quantity and unit, maps it to an exact catalog name (or *null* if we don't carry it), and records the deadline and intent. Code then converts units (for example, 1 ream = 500 sheets) and validates the catalog names.
2. **Orchestrator → Inventory Agent (ASSESS).** `assess_order_fulfillment` checks each line in turn:
   - Is the item in our catalog?
   - How many are on hand (`get_stock_level`)? Stock already promised to earlier lines of the same order is subtracted.
   - If there's a shortfall, when would the supplier deliver it (`get_supplier_delivery_date`)? One dispatch day is added for delivery to the customer.
   - Is that date on or before the customer's deadline?
   - Does the restocking cost fit the procurement budget? The budget is derived from `get_cash_balance`.

   The agent can also call `check_stock_level`, `get_inventory_snapshot` and `estimate_supplier_delivery` for ad-hoc questions.
3. **Orchestrator → Quoting Agent.** It looks up comparable past quotes (`search_quote_history`). Then `calculate_quote` prices only the lines that can be fulfilled:
   - List price = unit cost × 1.25.
   - Bulk discount on the whole order: 5% from 500 units, 10% from 2,000, 15% from 10,000.
   - Totals above $20 are rounded down to the whole dollar as a courtesy.
   - A hard floor stops any quote from selling below cost × 1.05.

   The agent then writes a short explanation of the price for the customer.
4. **Orchestrator → Sales Agent.** It calls `check_cash_balance`. For orders above $1,000, or any order that needs restocking, it must also call `run_financial_health_check`, which uses `generate_financial_report` to project cash against a $5,000 reserve. It then calls `finalize_sale`, which records one `sales` row per line through `create_transaction` and issues an order reference. If the checks fail, it calls `decline_sale`.
5. **Sales → Inventory hand-off (PROCURE).** After a confirmed sale, the Inventory Agent buys the back-ordered shortfall (`procure_backordered_items`, which records `stock_orders`). It then applies a reorder-point policy (`replenish_low_stock_items`): any regular item below its `min_stock_level` is topped up to 3× that minimum, provided the purchase doesn't breach the cash reserve.
6. The **Orchestrator** writes the reply. An output validator rejects the reply and makes the agent retry if any of these are true:
   - A step of the procedure was skipped.
   - The order reference, a unit price with its basis ("per sheet" or "per unit"), a line subtotal, the bulk discount amount or the exact total is missing.
   - An item ordered in reams, boxes or packs is missing its unit conversion (for example, "500 reams x 500 sheets per ream = 250,000 sheets") or its equivalent package price ("$25.00 per ream").
   - The reply contains internal terms (cost, margin, markup, cash balance and similar) or placeholders such as `[Your Name]`.
7. After all requests, the **Business Advisor** reads the outcome log, the financial report (`generate_financial_report`) and the inventory snapshot (`get_all_inventory`). It writes `business_advisor_report.md`.

Every starter helper is used by at least one tool:

| Helper | Used by |
|---|---|
| `create_transaction` | Inventory (stock orders), Sales (sales) |
| `get_all_inventory` | Inventory, Business Advisor |
| `get_stock_level` | Inventory |
| `get_supplier_delivery_date` | Inventory |
| `get_cash_balance` | Sales, Inventory (budget), Business Advisor |
| `generate_financial_report` | Sales (health check), Business Advisor |
| `search_quote_history` | Quoting |

### 1.3 Decisions behind the architecture

- **LLMs reason; code calculates.** gpt-4o-mini is good at reading messy requests ("500 reams of printer paper", "poster boards 24x36") and at explaining outcomes, but unreliable at arithmetic and bookkeeping. So every number comes from deterministic tools built on the starter helpers: stock, lead times, prices, discounts and ledger rows. The agents choose which tools to call, interpret the results, decide (for example, whether to approve or decline a sale) and communicate. This is why the customer-facing totals in `test_results.csv` match the ledger exactly.
- **Shared, typed state instead of passing numbers in text.** Each request gets an `OrderContext`, which pydantic-ai passes as `deps` to every agent. Order lines, the assessment, the quote and the sale record move between agents as typed objects, so no agent can retype or alter another agent's numbers.
- **Separate responsibilities and ledger writes.** Only the Inventory Agent buys stock and only the Sales Agent books sales. That gives a clear audit trail and matches how a real company splits purchasing from sales.
- **Guardrails around the LLM.** Delegation tools refuse to run out of order (for example, no quote before the stock check). Tool-level checks block unsafe actions: no sale without a health check on large orders, and no sale that would breach the cash reserve. Code also confirms that back-orders were actually purchased, and the output validator enforces transparency and confidentiality. Transient API failures are retried only when nothing has been written to the ledger yet, so a sale can never be booked twice. If an agent fails completely, the customer gets a safe, deterministic reply instead of an error message.
- **Why pydantic-ai.** Typed tool signatures, structured inputs (`list[RequestedItem]`), dependency injection, output validators that trigger retries, and `FunctionModel`. With `FunctionModel` the whole pipeline could be tested offline with a scripted model before spending API calls.
- **Why use the fifth agent as a Business Advisor.** The four core roles fit in four agents. The fifth slot went to the rubric's "stand out" Business Advisor because it adds management value without blurring the core responsibilities. The live, colour-coded terminal trace of hand-offs and tool calls (`WorkflowTracer`) implements the "terminal animation" suggestion.

---

## 2. Evaluation results (`test_results.csv`)

All 20 requests in `quote_requests_sample.csv` were processed in a single run, with no errors and no fallback replies.

| Metric | Result |
|---|---|
| Requests that changed the cash balance | **18 / 20** |
| Requests fully fulfilled | **7** (requests 1, 4, 5, 6, 10, 11, 12) |
| Requests partially fulfilled (some lines confirmed, others refused with reasons) | **11** (2, 3, 7, 8, 9, 13, 14, 16, 17, 19, 20) |
| Requests not fulfilled, with reasons given | **2** (15, 18) |
| Order lines fulfilled | 36 of 60 (60%) |
| Lines not fulfilled | 12 not carried, 12 could not meet the deadline |
| Sales revenue | $15,002.50 from 36 sales transactions |
| Gross profit on goods sold | $1,172.00 |
| Cash, 1 Apr → 17 Apr | $45,059.70 → $46,686.90 (**+$1,627.20**) |
| Inventory value | $4,940.30 → $4,485.10 (stock sold and partly replenished) |
| Supplier purchases | 38 `stock_orders` transactions ($13,375.30): back-orders and replenishment |

**Rubric checks:**
- At least three requests changed cash: 18 did.
- At least three quotes were fulfilled: 18 were fully or partially fulfilled.
- Not every request was fulfilled: requests 15 and 18 were refused outright, and 24 individual lines were refused across other orders. Every refusal comes with a reason.

### 2.1 Strengths shown by the results

1. **Impossible constraints are recognised and explained.**
   - Request 15 (12 Apr: 10,000 A4 sheets by 15 Apr) needs a 7-day supplier lead time, so it can't arrive before 20 Apr. The customer is told this and offered the units already on hand. A3 paper and cardboard are explained as not part of the product range.
   - Request 18 (14 Apr, needed by 15 Apr) is refused line by line. Each line gets the earliest possible date and a reduced-quantity alternative.
2. **Partial fulfilment instead of all-or-nothing.** Customers get whatever can be delivered on time. Each refused line has its own reason: "not part of our product range" for balloons, A3, A5 and tickets, or "would only reach you by X, after your requested date of Y".
3. **Accurate, transparent pricing.** Every confirmed order lists each unit price with its basis ("per sheet" or "per unit"), each line subtotal as a calculation, the bulk-discount tier and amount, the courtesy rounding, the total and the delivery date.
   - Request 3 asked for "500 reams of printer paper". The reply shows the conversion and the price basis so the customer can check the figure: "500 reams x 500 sheets per ream = 250,000 sheets", "$0.05 per sheet (equivalent to $25.00 per ream)" and "250,000 sheets x $0.05 per sheet = $12,500.00". With the 10,000 A4 sheets, the order earned the 15% tier: $13,125.00 − $1,968.75 (15%) − $0.25 courtesy rounding = $11,156.00.
4. **Back-order and replenishment logic.** Large orders such as request 3 were met through supplier back-orders within the deadline. Each sale triggered a reorder-point check, so regular items didn't sit below their minimum level. The financial health check ran before every large or restock order.
5. **Confidentiality.** No reply mentions costs, margins, the cash balance or internal errors. An automated scan of all 20 responses found no internal terms and no placeholders.
6. **Resilience.** An earlier attempt hit a temporary HTTP 404 from the API proxy. That led to retry logic that only restarts a request when nothing has been written to the ledger, plus a safe fallback reply. The final run completed cleanly.

### 2.2 Areas for improvement found in the results

- **Thin margins on back-ordered goods.** Restocked items are sold at only about 6–25% above cost. Request 3, for example, brought in $11,156 of revenue for roughly $10,500 of procurement. Cash grew by only $1,627 across 18 sales.
- **Lead times ignore stock already on order.** A supplier order counts as stock from the moment it is placed, and replenishment orders are dated the day they are placed. The customer delivery date does respect the supplier lead time, but a later request may be promised stock that is still in transit.
- **Item mapping is decided by the LLM.** Small inconsistencies remain. "Poster boards (24×36)" was mapped to *Large poster paper* in request 7, but treated as not carried in request 16.
- **Upsell hints are not guaranteed.** The quote data includes the next discount tier, but whether the reply mentions it is left to the LLM. In this run, no reply did.
- **Some replies reveal stock counts.** For lines past the deadline, replies say how many units are on hand (for example, "405 sheets"). That helps the customer but discloses inventory levels.

---

## 3. Suggestions for further improvement

1. **Model in-transit inventory and real delivery dates.**
   - Record supplier orders with their arrival date (from `get_supplier_delivery_date`), and keep a separate "on order" quantity.
   - Assess availability against "on hand + arriving before the deadline" rather than "everything ordered".
   - Book cash on the actual payment or delivery date.

   This would stop over-promising, and it would let the system plan replenishment ahead of demand (for example, keeping A4 and cardstock above a dynamic safety stock, which would have saved requests 17 and 18).
2. **Margin-aware, history-calibrated pricing.**
   - Price back-ordered lines with a smaller discount than in-stock lines, or cap the discount relative to the supplier cost of the shortfall.
   - Calibrate the discount tiers from `quotes.csv`. Historical quotes show the discounts customers accepted for each order size and event type, and that could drive a recommended discount band instead of fixed tiers.
3. **Deterministic product matching.** Keep a synonym table for the catalog ("poster board" → *Large poster paper (24x36 inches)*, "printer paper" → *Standard copy paper*, "kraft envelopes" → *Envelopes*). Code would apply it before the LLM proposes a match, and the LLM would only handle unseen phrases. This removes the inconsistencies noted above and makes "not carried" decisions repeatable.
4. **A customer negotiation agent and multi-turn confirmation.** Today each request is handled in one turn and feasible lines are booked straight away. A customer-side agent, or a confirmation step, would let customers accept reduced quantities, later dates or substitutes before booking. That converts more of the 24 lost lines into revenue.
5. **Act on the Business Advisor's findings.** The advisor reported that lost demand comes mostly from items we don't carry (A3/A5 formats, cardboard, tickets) and from short lead times. Adding A3 paper to the range and keeping a safety stock of fast movers (copy paper, A4, cardstock) are concrete next steps the system could adopt automatically.

---

## 4. Industry best practices

- **Transparent, explainable outputs.** Customer replies include everything relevant to the request:
  - per-item status and delivery date
  - unit prices with their basis (per sheet or per unit), and each line subtotal shown as a calculation
  - for items ordered in reams, boxes or packs, the conversion to sheets and the equivalent package price
  - the discount tier, why it applies and how it was calculated, and the total
  - a reason for every item that can't be supplied

  The output validator enforces the order reference, every unit price with its basis, every unit conversion, the line subtotals, the discount amount and the exact total.
- **No sensitive information.** Customers never see costs, margins, markup, the cash balance, supplier details or internal errors. The validator blocks such terms, the data sent to the orchestrator is a customer-safe view, and exceptions are logged internally while the customer gets a polite fallback. No customer personal data is stored or repeated.
- **Readable, modular code.** `project_starter.py` contains the unchanged starter helpers followed by clearly separated sections: configuration and business policy, tracing, data models, deterministic domain services, agent prompts, agent definitions, per-agent tools, the multi-agent system class, and the evaluation harness. Naming is consistent (snake_case for functions, PascalCase for classes, UPPER_CASE for policy constants). Every tool and service has a docstring, and comments explain the business rules.

---

## 5. How to reproduce

```bash
pip install -r requirements.txt
cp .env.example .env          # set UDACITY_OPENAI_API_KEY
python project_starter.py     # writes test_results.csv and business_advisor_report.md
```
