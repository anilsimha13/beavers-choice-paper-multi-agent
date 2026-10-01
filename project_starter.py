import pandas as pd
import numpy as np
import os
import time
import dotenv
import ast
from sqlalchemy.sql import text
from datetime import datetime, timedelta
from typing import Dict, List, Union
from sqlalchemy import create_engine, Engine

# Create an SQLite database
db_engine = create_engine("sqlite:///munder_difflin.db")

# List containing the different kinds of papers 
paper_supplies = [
    # Paper Types (priced per sheet unless specified)
    {"item_name": "A4 paper",                         "category": "paper",        "unit_price": 0.05},
    {"item_name": "Letter-sized paper",              "category": "paper",        "unit_price": 0.06},
    {"item_name": "Cardstock",                        "category": "paper",        "unit_price": 0.15},
    {"item_name": "Colored paper",                    "category": "paper",        "unit_price": 0.10},
    {"item_name": "Glossy paper",                     "category": "paper",        "unit_price": 0.20},
    {"item_name": "Matte paper",                      "category": "paper",        "unit_price": 0.18},
    {"item_name": "Recycled paper",                   "category": "paper",        "unit_price": 0.08},
    {"item_name": "Eco-friendly paper",               "category": "paper",        "unit_price": 0.12},
    {"item_name": "Poster paper",                     "category": "paper",        "unit_price": 0.25},
    {"item_name": "Banner paper",                     "category": "paper",        "unit_price": 0.30},
    {"item_name": "Kraft paper",                      "category": "paper",        "unit_price": 0.10},
    {"item_name": "Construction paper",               "category": "paper",        "unit_price": 0.07},
    {"item_name": "Wrapping paper",                   "category": "paper",        "unit_price": 0.15},
    {"item_name": "Glitter paper",                    "category": "paper",        "unit_price": 0.22},
    {"item_name": "Decorative paper",                 "category": "paper",        "unit_price": 0.18},
    {"item_name": "Letterhead paper",                 "category": "paper",        "unit_price": 0.12},
    {"item_name": "Legal-size paper",                 "category": "paper",        "unit_price": 0.08},
    {"item_name": "Crepe paper",                      "category": "paper",        "unit_price": 0.05},
    {"item_name": "Photo paper",                      "category": "paper",        "unit_price": 0.25},
    {"item_name": "Uncoated paper",                   "category": "paper",        "unit_price": 0.06},
    {"item_name": "Butcher paper",                    "category": "paper",        "unit_price": 0.10},
    {"item_name": "Heavyweight paper",                "category": "paper",        "unit_price": 0.20},
    {"item_name": "Standard copy paper",              "category": "paper",        "unit_price": 0.04},
    {"item_name": "Bright-colored paper",             "category": "paper",        "unit_price": 0.12},
    {"item_name": "Patterned paper",                  "category": "paper",        "unit_price": 0.15},

    # Product Types (priced per unit)
    {"item_name": "Paper plates",                     "category": "product",      "unit_price": 0.10},  # per plate
    {"item_name": "Paper cups",                       "category": "product",      "unit_price": 0.08},  # per cup
    {"item_name": "Paper napkins",                    "category": "product",      "unit_price": 0.02},  # per napkin
    {"item_name": "Disposable cups",                  "category": "product",      "unit_price": 0.10},  # per cup
    {"item_name": "Table covers",                     "category": "product",      "unit_price": 1.50},  # per cover
    {"item_name": "Envelopes",                        "category": "product",      "unit_price": 0.05},  # per envelope
    {"item_name": "Sticky notes",                     "category": "product",      "unit_price": 0.03},  # per sheet
    {"item_name": "Notepads",                         "category": "product",      "unit_price": 2.00},  # per pad
    {"item_name": "Invitation cards",                 "category": "product",      "unit_price": 0.50},  # per card
    {"item_name": "Flyers",                           "category": "product",      "unit_price": 0.15},  # per flyer
    {"item_name": "Party streamers",                  "category": "product",      "unit_price": 0.05},  # per roll
    {"item_name": "Decorative adhesive tape (washi tape)", "category": "product", "unit_price": 0.20},  # per roll
    {"item_name": "Paper party bags",                 "category": "product",      "unit_price": 0.25},  # per bag
    {"item_name": "Name tags with lanyards",          "category": "product",      "unit_price": 0.75},  # per tag
    {"item_name": "Presentation folders",             "category": "product",      "unit_price": 0.50},  # per folder

    # Large-format items (priced per unit)
    {"item_name": "Large poster paper (24x36 inches)", "category": "large_format", "unit_price": 1.00},
    {"item_name": "Rolls of banner paper (36-inch width)", "category": "large_format", "unit_price": 2.50},

    # Specialty papers
    {"item_name": "100 lb cover stock",               "category": "specialty",    "unit_price": 0.50},
    {"item_name": "80 lb text paper",                 "category": "specialty",    "unit_price": 0.40},
    {"item_name": "250 gsm cardstock",                "category": "specialty",    "unit_price": 0.30},
    {"item_name": "220 gsm poster paper",             "category": "specialty",    "unit_price": 0.35},
]

# Given below are some utility functions you can use to implement your multi-agent system

def generate_sample_inventory(paper_supplies: list, coverage: float = 0.4, seed: int = 137) -> pd.DataFrame:
    """
    Generate inventory for exactly a specified percentage of items from the full paper supply list.

    This function randomly selects exactly `coverage` × N items from the `paper_supplies` list,
    and assigns each selected item:
    - a random stock quantity between 200 and 800,
    - a minimum stock level between 50 and 150.

    The random seed ensures reproducibility of selection and stock levels.

    Args:
        paper_supplies (list): A list of dictionaries, each representing a paper item with
                               keys 'item_name', 'category', and 'unit_price'.
        coverage (float, optional): Fraction of items to include in the inventory (default is 0.4, or 40%).
        seed (int, optional): Random seed for reproducibility (default is 137).

    Returns:
        pd.DataFrame: A DataFrame with the selected items and assigned inventory values, including:
                      - item_name
                      - category
                      - unit_price
                      - current_stock
                      - min_stock_level
    """
    # Ensure reproducible random output
    np.random.seed(seed)

    # Calculate number of items to include based on coverage
    num_items = int(len(paper_supplies) * coverage)

    # Randomly select item indices without replacement
    selected_indices = np.random.choice(
        range(len(paper_supplies)),
        size=num_items,
        replace=False
    )

    # Extract selected items from paper_supplies list
    selected_items = [paper_supplies[i] for i in selected_indices]

    # Construct inventory records
    inventory = []
    for item in selected_items:
        inventory.append({
            "item_name": item["item_name"],
            "category": item["category"],
            "unit_price": item["unit_price"],
            "current_stock": np.random.randint(200, 800),  # Realistic stock range
            "min_stock_level": np.random.randint(50, 150)  # Reasonable threshold for reordering
        })

    # Return inventory as a pandas DataFrame
    return pd.DataFrame(inventory)

def init_database(db_engine: Engine, seed: int = 137) -> Engine:    
    """
    Set up the Munder Difflin database with all required tables and initial records.

    This function performs the following tasks:
    - Creates the 'transactions' table for logging stock orders and sales
    - Loads customer inquiries from 'quote_requests.csv' into a 'quote_requests' table
    - Loads previous quotes from 'quotes.csv' into a 'quotes' table, extracting useful metadata
    - Generates a random subset of paper inventory using `generate_sample_inventory`
    - Inserts initial financial records including available cash and starting stock levels

    Args:
        db_engine (Engine): A SQLAlchemy engine connected to the SQLite database.
        seed (int, optional): A random seed used to control reproducibility of inventory stock levels.
                              Default is 137.

    Returns:
        Engine: The same SQLAlchemy engine, after initializing all necessary tables and records.

    Raises:
        Exception: If an error occurs during setup, the exception is printed and raised.
    """
    try:
        # ----------------------------
        # 1. Create an empty 'transactions' table schema
        # ----------------------------
        transactions_schema = pd.DataFrame({
            "id": [],
            "item_name": [],
            "transaction_type": [],  # 'stock_orders' or 'sales'
            "units": [],             # Quantity involved
            "price": [],             # Total price for the transaction
            "transaction_date": [],  # ISO-formatted date
        })
        transactions_schema.to_sql("transactions", db_engine, if_exists="replace", index=False)

        # Set a consistent starting date
        initial_date = datetime(2025, 1, 1).isoformat()

        # ----------------------------
        # 2. Load and initialize 'quote_requests' table
        # ----------------------------
        quote_requests_df = pd.read_csv("quote_requests.csv")
        quote_requests_df["id"] = range(1, len(quote_requests_df) + 1)
        quote_requests_df.to_sql("quote_requests", db_engine, if_exists="replace", index=False)

        # ----------------------------
        # 3. Load and transform 'quotes' table
        # ----------------------------
        quotes_df = pd.read_csv("quotes.csv")
        quotes_df["request_id"] = range(1, len(quotes_df) + 1)
        quotes_df["order_date"] = initial_date

        # Unpack metadata fields (job_type, order_size, event_type) if present
        if "request_metadata" in quotes_df.columns:
            quotes_df["request_metadata"] = quotes_df["request_metadata"].apply(
                lambda x: ast.literal_eval(x) if isinstance(x, str) else x
            )
            quotes_df["job_type"] = quotes_df["request_metadata"].apply(lambda x: x.get("job_type", ""))
            quotes_df["order_size"] = quotes_df["request_metadata"].apply(lambda x: x.get("order_size", ""))
            quotes_df["event_type"] = quotes_df["request_metadata"].apply(lambda x: x.get("event_type", ""))

        # Retain only relevant columns
        quotes_df = quotes_df[[
            "request_id",
            "total_amount",
            "quote_explanation",
            "order_date",
            "job_type",
            "order_size",
            "event_type"
        ]]
        quotes_df.to_sql("quotes", db_engine, if_exists="replace", index=False)

        # ----------------------------
        # 4. Generate inventory and seed stock
        # ----------------------------
        inventory_df = generate_sample_inventory(paper_supplies, seed=seed)

        # Seed initial transactions
        initial_transactions = []

        # Add a starting cash balance via a dummy sales transaction
        initial_transactions.append({
            "item_name": None,
            "transaction_type": "sales",
            "units": None,
            "price": 50000.0,
            "transaction_date": initial_date,
        })

        # Add one stock order transaction per inventory item
        for _, item in inventory_df.iterrows():
            initial_transactions.append({
                "item_name": item["item_name"],
                "transaction_type": "stock_orders",
                "units": item["current_stock"],
                "price": item["current_stock"] * item["unit_price"],
                "transaction_date": initial_date,
            })

        # Commit transactions to database
        pd.DataFrame(initial_transactions).to_sql("transactions", db_engine, if_exists="append", index=False)

        # Save the inventory reference table
        inventory_df.to_sql("inventory", db_engine, if_exists="replace", index=False)

        return db_engine

    except Exception as e:
        print(f"Error initializing database: {e}")
        raise

def create_transaction(
    item_name: str,
    transaction_type: str,
    quantity: int,
    price: float,
    date: Union[str, datetime],
) -> int:
    """
    This function records a transaction of type 'stock_orders' or 'sales' with a specified
    item name, quantity, total price, and transaction date into the 'transactions' table of the database.

    Args:
        item_name (str): The name of the item involved in the transaction.
        transaction_type (str): Either 'stock_orders' or 'sales'.
        quantity (int): Number of units involved in the transaction.
        price (float): Total price of the transaction.
        date (str or datetime): Date of the transaction in ISO 8601 format.

    Returns:
        int: The ID of the newly inserted transaction.

    Raises:
        ValueError: If `transaction_type` is not 'stock_orders' or 'sales'.
        Exception: For other database or execution errors.
    """
    try:
        # Convert datetime to ISO string if necessary
        date_str = date.isoformat() if isinstance(date, datetime) else date

        # Validate transaction type
        if transaction_type not in {"stock_orders", "sales"}:
            raise ValueError("Transaction type must be 'stock_orders' or 'sales'")

        # Prepare transaction record as a single-row DataFrame
        transaction = pd.DataFrame([{
            "item_name": item_name,
            "transaction_type": transaction_type,
            "units": quantity,
            "price": price,
            "transaction_date": date_str,
        }])

        # Insert the record into the database
        transaction.to_sql("transactions", db_engine, if_exists="append", index=False)

        # Fetch and return the ID of the inserted row
        result = pd.read_sql("SELECT last_insert_rowid() as id", db_engine)
        return int(result.iloc[0]["id"])

    except Exception as e:
        print(f"Error creating transaction: {e}")
        raise

def get_all_inventory(as_of_date: str) -> Dict[str, int]:
    """
    Retrieve a snapshot of available inventory as of a specific date.

    This function calculates the net quantity of each item by summing 
    all stock orders and subtracting all sales up to and including the given date.

    Only items with positive stock are included in the result.

    Args:
        as_of_date (str): ISO-formatted date string (YYYY-MM-DD) representing the inventory cutoff.

    Returns:
        Dict[str, int]: A dictionary mapping item names to their current stock levels.
    """
    # SQL query to compute stock levels per item as of the given date
    query = """
        SELECT
            item_name,
            SUM(CASE
                WHEN transaction_type = 'stock_orders' THEN units
                WHEN transaction_type = 'sales' THEN -units
                ELSE 0
            END) as stock
        FROM transactions
        WHERE item_name IS NOT NULL
        AND transaction_date <= :as_of_date
        GROUP BY item_name
        HAVING stock > 0
    """

    # Execute the query with the date parameter
    result = pd.read_sql(query, db_engine, params={"as_of_date": as_of_date})

    # Convert the result into a dictionary {item_name: stock}
    return dict(zip(result["item_name"], result["stock"]))

def get_stock_level(item_name: str, as_of_date: Union[str, datetime]) -> pd.DataFrame:
    """
    Retrieve the stock level of a specific item as of a given date.

    This function calculates the net stock by summing all 'stock_orders' and 
    subtracting all 'sales' transactions for the specified item up to the given date.

    Args:
        item_name (str): The name of the item to look up.
        as_of_date (str or datetime): The cutoff date (inclusive) for calculating stock.

    Returns:
        pd.DataFrame: A single-row DataFrame with columns 'item_name' and 'current_stock'.
    """
    # Convert date to ISO string format if it's a datetime object
    if isinstance(as_of_date, datetime):
        as_of_date = as_of_date.isoformat()

    # SQL query to compute net stock level for the item
    stock_query = """
        SELECT
            item_name,
            COALESCE(SUM(CASE
                WHEN transaction_type = 'stock_orders' THEN units
                WHEN transaction_type = 'sales' THEN -units
                ELSE 0
            END), 0) AS current_stock
        FROM transactions
        WHERE item_name = :item_name
        AND transaction_date <= :as_of_date
    """

    # Execute query and return result as a DataFrame
    return pd.read_sql(
        stock_query,
        db_engine,
        params={"item_name": item_name, "as_of_date": as_of_date},
    )

def get_supplier_delivery_date(input_date_str: str, quantity: int) -> str:
    """
    Estimate the supplier delivery date based on the requested order quantity and a starting date.

    Delivery lead time increases with order size:
        - ≤10 units: same day
        - 11–100 units: 1 day
        - 101–1000 units: 4 days
        - >1000 units: 7 days

    Args:
        input_date_str (str): The starting date in ISO format (YYYY-MM-DD).
        quantity (int): The number of units in the order.

    Returns:
        str: Estimated delivery date in ISO format (YYYY-MM-DD).
    """
    # Debug log (comment out in production if needed)
    print(f"FUNC (get_supplier_delivery_date): Calculating for qty {quantity} from date string '{input_date_str}'")

    # Attempt to parse the input date
    try:
        input_date_dt = datetime.fromisoformat(input_date_str.split("T")[0])
    except (ValueError, TypeError):
        # Fallback to current date on format error
        print(f"WARN (get_supplier_delivery_date): Invalid date format '{input_date_str}', using today as base.")
        input_date_dt = datetime.now()

    # Determine delivery delay based on quantity
    if quantity <= 10:
        days = 0
    elif quantity <= 100:
        days = 1
    elif quantity <= 1000:
        days = 4
    else:
        days = 7

    # Add delivery days to the starting date
    delivery_date_dt = input_date_dt + timedelta(days=days)

    # Return formatted delivery date
    return delivery_date_dt.strftime("%Y-%m-%d")

def get_cash_balance(as_of_date: Union[str, datetime]) -> float:
    """
    Calculate the current cash balance as of a specified date.

    The balance is computed by subtracting total stock purchase costs ('stock_orders')
    from total revenue ('sales') recorded in the transactions table up to the given date.

    Args:
        as_of_date (str or datetime): The cutoff date (inclusive) in ISO format or as a datetime object.

    Returns:
        float: Net cash balance as of the given date. Returns 0.0 if no transactions exist or an error occurs.
    """
    try:
        # Convert date to ISO format if it's a datetime object
        if isinstance(as_of_date, datetime):
            as_of_date = as_of_date.isoformat()

        # Query all transactions on or before the specified date
        transactions = pd.read_sql(
            "SELECT * FROM transactions WHERE transaction_date <= :as_of_date",
            db_engine,
            params={"as_of_date": as_of_date},
        )

        # Compute the difference between sales and stock purchases
        if not transactions.empty:
            total_sales = transactions.loc[transactions["transaction_type"] == "sales", "price"].sum()
            total_purchases = transactions.loc[transactions["transaction_type"] == "stock_orders", "price"].sum()
            return float(total_sales - total_purchases)

        return 0.0

    except Exception as e:
        print(f"Error getting cash balance: {e}")
        return 0.0


def generate_financial_report(as_of_date: Union[str, datetime]) -> Dict:
    """
    Generate a complete financial report for the company as of a specific date.

    This includes:
    - Cash balance
    - Inventory valuation
    - Combined asset total
    - Itemized inventory breakdown
    - Top 5 best-selling products

    Args:
        as_of_date (str or datetime): The date (inclusive) for which to generate the report.

    Returns:
        Dict: A dictionary containing the financial report fields:
            - 'as_of_date': The date of the report
            - 'cash_balance': Total cash available
            - 'inventory_value': Total value of inventory
            - 'total_assets': Combined cash and inventory value
            - 'inventory_summary': List of items with stock and valuation details
            - 'top_selling_products': List of top 5 products by revenue
    """
    # Normalize date input
    if isinstance(as_of_date, datetime):
        as_of_date = as_of_date.isoformat()

    # Get current cash balance
    cash = get_cash_balance(as_of_date)

    # Get current inventory snapshot
    inventory_df = pd.read_sql("SELECT * FROM inventory", db_engine)
    inventory_value = 0.0
    inventory_summary = []

    # Compute total inventory value and summary by item
    for _, item in inventory_df.iterrows():
        stock_info = get_stock_level(item["item_name"], as_of_date)
        stock = stock_info["current_stock"].iloc[0]
        item_value = stock * item["unit_price"]
        inventory_value += item_value

        inventory_summary.append({
            "item_name": item["item_name"],
            "stock": stock,
            "unit_price": item["unit_price"],
            "value": item_value,
        })

    # Identify top-selling products by revenue
    top_sales_query = """
        SELECT item_name, SUM(units) as total_units, SUM(price) as total_revenue
        FROM transactions
        WHERE transaction_type = 'sales' AND transaction_date <= :date
        GROUP BY item_name
        ORDER BY total_revenue DESC
        LIMIT 5
    """
    top_sales = pd.read_sql(top_sales_query, db_engine, params={"date": as_of_date})
    top_selling_products = top_sales.to_dict(orient="records")

    return {
        "as_of_date": as_of_date,
        "cash_balance": cash,
        "inventory_value": inventory_value,
        "total_assets": cash + inventory_value,
        "inventory_summary": inventory_summary,
        "top_selling_products": top_selling_products,
    }


def search_quote_history(search_terms: List[str], limit: int = 5) -> List[Dict]:
    """
    Retrieve a list of historical quotes that match any of the provided search terms.

    The function searches both the original customer request (from `quote_requests`) and
    the explanation for the quote (from `quotes`) for each keyword. Results are sorted by
    most recent order date and limited by the `limit` parameter.

    Args:
        search_terms (List[str]): List of terms to match against customer requests and explanations.
        limit (int, optional): Maximum number of quote records to return. Default is 5.

    Returns:
        List[Dict]: A list of matching quotes, each represented as a dictionary with fields:
            - original_request
            - total_amount
            - quote_explanation
            - job_type
            - order_size
            - event_type
            - order_date
    """
    conditions = []
    params = {}

    # Build SQL WHERE clause using LIKE filters for each search term
    for i, term in enumerate(search_terms):
        param_name = f"term_{i}"
        conditions.append(
            f"(LOWER(qr.response) LIKE :{param_name} OR "
            f"LOWER(q.quote_explanation) LIKE :{param_name})"
        )
        params[param_name] = f"%{term.lower()}%"

    # Combine conditions; fallback to always-true if no terms provided
    where_clause = " AND ".join(conditions) if conditions else "1=1"

    # Final SQL query to join quotes with quote_requests
    query = f"""
        SELECT
            qr.response AS original_request,
            q.total_amount,
            q.quote_explanation,
            q.job_type,
            q.order_size,
            q.event_type,
            q.order_date
        FROM quotes q
        JOIN quote_requests qr ON q.request_id = qr.id
        WHERE {where_clause}
        ORDER BY q.order_date DESC
        LIMIT {limit}
    """

    # Execute parameterized query
    with db_engine.connect() as conn:
        result = conn.execute(text(query), params)
        return [dict(row._mapping) for row in result]

########################
########################
########################
# YOUR MULTI AGENT STARTS HERE
########################
########################
########################

# =============================================================================
# Beaver's Choice Paper Company - Multi-Agent Quoting, Inventory & Sales System
# =============================================================================
#
# Framework: pydantic-ai
#
# Agents (5 in total, the maximum allowed):
#   1. Orchestrator Agent     - customer-facing; understands the request, delegates
#                               work to the workers and writes the final reply.
#   2. Inventory Agent        - stock checks, supplier lead times, reorder decisions,
#                               procurement of back-ordered items and replenishment.
#   3. Quoting Agent          - pricing with bulk discounts, informed by quote history.
#   4. Sales Agent            - financial health check and booking of the sale.
#   5. Business Advisor Agent - post-run analysis of all transactions with
#                               recommendations to improve efficiency and revenue.
#
# Design principle: LLM agents decide *what* to do and *explain* it, while every
# number that matters (stock, prices, discounts, ledger entries) is produced by
# deterministic tools built on the starter helper functions. This keeps the
# customer-facing figures accurate and auditable.
#
# Helper function -> tool mapping:
#   get_stock_level            -> Inventory: check_stock_level, assess_order_fulfillment
#   get_all_inventory          -> Inventory: get_inventory_snapshot, replenish_low_stock_items
#                                 Advisor:   get_inventory_snapshot_for_advisor
#   get_supplier_delivery_date -> Inventory: estimate_supplier_delivery, assess_order_fulfillment
#   create_transaction         -> Inventory: procure_backordered_items, replenish_low_stock_items
#                                 Sales:     finalize_sale
#   get_cash_balance           -> Sales:     check_cash_balance
#                                 (also used inside assess_order_fulfillment for budgeting)
#   generate_financial_report  -> Sales:     run_financial_health_check
#                                 Advisor:   get_financial_report
#   search_quote_history       -> Quoting:   find_similar_past_quotes
# =============================================================================

import difflib
import json
import math
import re
import sys
from dataclasses import dataclass, field
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field

os.environ.setdefault("PYDANTIC_AI_NO_BANNER", "1")  # keep the console trace clean
from pydantic_ai import Agent, ModelRetry, RunContext
from pydantic_ai.providers.openai import OpenAIProvider

try:  # pydantic-ai >= 1.0
    from pydantic_ai.models.openai import OpenAIChatModel as _OpenAIModelClass
except ImportError:  # older pydantic-ai releases
    from pydantic_ai.models.openai import OpenAIModel as _OpenAIModelClass

dotenv.load_dotenv()

# Make console output robust on Windows terminals (ANSI colours + non-ASCII text).
if os.name == "nt":
    os.system("")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(errors="replace")


# -----------------------------------------------------------------------------
# LLM configuration
# -----------------------------------------------------------------------------
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://openai.vocareum.com/v1")
OPENAI_MODEL_NAME = os.getenv("OPENAI_MODEL_NAME", "gpt-4o-mini")
OPENAI_API_KEY = os.getenv("UDACITY_OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")

llm_model = _OpenAIModelClass(
    OPENAI_MODEL_NAME,
    provider=OpenAIProvider(
        base_url=OPENAI_BASE_URL,
        # A placeholder keeps imports working without a key; calls will then fail
        # and are handled gracefully by the workflow.
        api_key=OPENAI_API_KEY or "missing-api-key",
    ),
)

# Low temperature for consistent business behaviour; sequential tool calls so the
# orchestrator never runs dependent steps (e.g. quote before stock check) in parallel.
AGENT_MODEL_SETTINGS = {"temperature": 0.1, "parallel_tool_calls": False}


# -----------------------------------------------------------------------------
# Business policy (internal - never shown to customers)
# -----------------------------------------------------------------------------
# `unit_price` in the starter data is what the company pays its supplier: it is
# the price used when stock is purchased (see init_database) and when inventory
# is valued (see generate_financial_report). Customers pay a list price on top.
LIST_PRICE_MARKUP = 1.25
MINIMUM_MARGIN_MULTIPLIER = 1.05      # a quote may never fall below cost x this
BULK_DISCOUNT_TIERS = [               # (minimum total units in the order, discount)
    (10_000, 0.15),
    (2_000, 0.10),
    (500, 0.05),
]
FRIENDLY_ROUNDING_THRESHOLD = 20.0    # totals above this are rounded down to whole dollars
IN_STOCK_DISPATCH_DAYS = 1            # warehouse -> customer delivery time
CASH_RESERVE = 5_000.0                # procurement may never push cash below this
MAX_ORDER_PROCUREMENT_SHARE = 0.5     # one order may use at most 50% of cash for restocking
LARGE_ORDER_THRESHOLD = 1_000.0       # quotes above this require a full financial report
MAX_REQUEST_ATTEMPTS = 3              # retries for transient LLM/API failures
RETRY_BACKOFF_SECONDS = 10
REORDER_UP_TO_MULTIPLIER = 3          # replenish stocked items up to 3x their minimum level

UNITS_PER_REAM = 500
DEFAULT_UNITS_PER_BOX = 5_000         # a case/box of paper = 10 reams
DEFAULT_UNITS_PER_PACK = 100
PACKAGE_UNITS = [                     # (keyword in the customer's unit, default units per package)
    ("ream", UNITS_PER_REAM),
    ("box", DEFAULT_UNITS_PER_BOX),
    ("case", DEFAULT_UNITS_PER_BOX),
    ("carton", DEFAULT_UNITS_PER_BOX),
    ("packet", DEFAULT_UNITS_PER_PACK),
    ("pack", DEFAULT_UNITS_PER_PACK),
]
SHEET_PRICED_CATEGORIES = {"paper", "specialty"}

CATALOG: dict[str, dict] = {item["item_name"]: item for item in paper_supplies}
CATALOG_BY_LOWER_NAME: dict[str, str] = {name.lower(): name for name in CATALOG}

# Phrases that must never appear in customer-facing replies.
FORBIDDEN_CUSTOMER_TERMS = [
    "cash balance", "markup", "mark-up", "margin", "profit", "unit cost",
    "our cost", "cost price", "supplier cost", "wholesale", "stock_orders",
    "transaction id", "traceback", "exception", "internal",
]

# Line statuses produced by the inventory assessment.
STATUS_IN_STOCK = "in_stock"
STATUS_BACKORDER = "backorder"
STATUS_NOT_CARRIED = "not_carried"
STATUS_MISSES_DEADLINE = "misses_deadline"
STATUS_OVER_CAPACITY = "over_capacity"
FULFILLABLE_STATUSES = {STATUS_IN_STOCK, STATUS_BACKORDER}

CUSTOMER_STATUS_LABELS = {
    STATUS_IN_STOCK: "available from stock",
    STATUS_BACKORDER: "sourced from our supplier",
    STATUS_NOT_CARRIED: "not available",
    STATUS_MISSES_DEADLINE: "cannot be delivered in time",
    STATUS_OVER_CAPACITY: "cannot be sourced in this volume",
}


# -----------------------------------------------------------------------------
# Terminal animation / tracing of the agent workflow
# -----------------------------------------------------------------------------
class WorkflowTracer:
    """Prints a live, colour-coded trace of agent hand-offs and tool calls.

    The trace shows the customer how their request moves through the team.
    Set the environment variable BCP_ANIMATION=0 to disable the typing effect
    and BCP_TRACE=0 to silence the trace completely.
    """

    AGENT_COLOURS = {
        "Customer": "\033[97m",
        "Orchestrator": "\033[95m",
        "Inventory Agent": "\033[94m",
        "Quoting Agent": "\033[93m",
        "Sales Agent": "\033[92m",
        "Business Advisor": "\033[96m",
    }
    RESET = "\033[0m"
    DIM = "\033[2m"

    def __init__(self) -> None:
        self.enabled = os.getenv("BCP_TRACE", "1") != "0"
        self.animate = (
            self.enabled
            and os.getenv("BCP_ANIMATION", "1") != "0"
            and sys.stdout.isatty()
        )

    def _colour(self, agent: str) -> str:
        return self.AGENT_COLOURS.get(agent, "")

    def _emit(self, text: str) -> None:
        if not self.enabled:
            return
        if self.animate:
            for char in text:
                sys.stdout.write(char)
                sys.stdout.flush()
                time.sleep(0.002)
            sys.stdout.write("\n")
        else:
            print(text)

    def handoff(self, source: str, target: str, message: str) -> None:
        """Show one agent delegating work to another."""
        self._emit(
            f"  {self._colour(source)}[{source}]{self.RESET} ==> "
            f"{self._colour(target)}[{target}]{self.RESET} {message}"
        )

    def tool(self, agent: str, tool_name: str, detail: str = "") -> None:
        """Show an agent invoking one of its tools."""
        suffix = f" {self.DIM}{detail}{self.RESET}" if detail else ""
        self._emit(f"      {self._colour(agent)}{agent}{self.RESET} -> {tool_name}(){suffix}")

    def note(self, agent: str, message: str) -> None:
        """Show an agent's conclusion or a workflow event."""
        self._emit(f"      {self._colour(agent)}{agent}{self.RESET}: {message}")


tracer = WorkflowTracer()


# -----------------------------------------------------------------------------
# Data models shared between agents
# -----------------------------------------------------------------------------
class RequestedItem(BaseModel):
    """One item extracted from the customer's request by the orchestrator."""

    customer_description: str = Field(description="The item as the customer described it, e.g. '200 sheets of A4 glossy paper'.")
    quantity: int = Field(gt=0, description="Number of units exactly as stated by the customer (before any unit conversion).")
    unit: str = Field(default="unit", description="Unit as stated by the customer: sheet, ream, roll, pack, packet, box, unit, ...")
    units_per_package: Optional[int] = Field(default=None, description="Units inside one pack/box/ream ONLY if the customer states it explicitly, otherwise null.")
    catalog_item: Optional[str] = Field(default=None, description="Exact name of the matching catalog item, or null if no catalog item genuinely matches.")


def pluralize(word: str, count: int) -> str:
    if count == 1:
        return word
    return word + "es" if word.endswith(("x", "s")) else word + "s"


class LineQuantity(BaseModel):
    """Quantity of an order line in pricing units, plus the customer's original package unit."""

    quantity: int                      # in pricing units (sheets / units)
    pricing_unit: str = "unit"         # "sheet" or "unit" - the basis of the unit price
    requested_quantity: int = 0        # as stated by the customer, e.g. 500 (reams)
    package_unit: str = ""             # "ream", "box", ... or "" when ordered in single units
    units_per_package: int = 1

    @property
    def quantity_text(self) -> str:
        return f"{self.quantity:,} {pluralize(self.pricing_unit, self.quantity)}"

    @property
    def quantity_note(self) -> str:
        """Conversion shown to the customer, e.g. '500 reams x 500 sheets per ream = 250,000 sheets'."""
        if not self.package_unit:
            return ""
        return (f"{self.requested_quantity:,} {pluralize(self.package_unit, self.requested_quantity)} x "
                f"{self.per_package_text} = {self.quantity_text}")

    @property
    def per_package_text(self) -> str:
        return f"{self.units_per_package:,} {pluralize(self.pricing_unit, self.units_per_package)} per {self.package_unit}"


class FulfillmentLine(LineQuantity):
    """An order line enriched with the inventory agent's assessment."""

    line_id: int
    customer_description: str
    catalog_item: Optional[str]
    unit_cost: float = 0.0             # internal
    is_stocked_item: bool = False      # part of the regular inventory range
    stock_on_hand: int = 0             # internal
    shortfall: int = 0                 # units that must be bought from the supplier
    supplier_eta: Optional[str] = None
    estimated_delivery: Optional[str] = None
    status: str = "pending"
    customer_reason: str = ""          # customer-safe explanation of the status
    procured: bool = False

    @property
    def is_fulfillable(self) -> bool:
        return self.status in FULFILLABLE_STATUSES


class QuoteLine(LineQuantity):
    catalog_item: str
    customer_description: str
    list_unit_price: float             # per pricing unit
    line_subtotal: float
    line_total: float                  # after discount and rounding, used for the sale record
    unit_cost: float                   # internal

    @property
    def unit_price_text(self) -> str:
        return f"{format_unit_price(self.list_unit_price)} per {self.pricing_unit}"

    @property
    def package_price_text(self) -> str:
        if not self.package_unit:
            return ""
        return f"{format_unit_price(self.list_unit_price * self.units_per_package)} per {self.package_unit}"

    @property
    def subtotal_calculation(self) -> str:
        return f"{self.quantity_text} x {self.unit_price_text} = {format_money(self.line_subtotal)}"


class Quote(BaseModel):
    lines: list[QuoteLine]
    total_units: int
    subtotal: float
    discount_rate: float
    discount_amount: float
    rounding_adjustment: float
    total: float
    procurement_cost: float            # internal: cost of back-ordered units
    goods_cost: float                  # internal: cost of all goods sold


class SaleRecord(BaseModel):
    order_reference: str
    total_charged: float
    transaction_ids: list[int]
    delivery_date: str


@dataclass
class OrderContext:
    """Shared state for one customer request, passed to every agent as deps."""

    request_id: int
    request_text: str
    request_date: str
    customer_intent: str = "order"
    delivery_deadline: Optional[str] = None
    lines: list[FulfillmentLine] = field(default_factory=list)
    items_recorded: bool = False
    assessment_done: bool = False
    quote: Optional[Quote] = None
    quote_explanation: str = ""
    similar_quotes_found: int = 0
    health_check_done: bool = False
    sale: Optional[SaleRecord] = None
    decline_reason: Optional[str] = None
    procurement_log: list[dict] = field(default_factory=list)
    replenishment_log: list[dict] = field(default_factory=list)
    replenishment_done: bool = False

    @property
    def fulfillable_lines(self) -> list[FulfillmentLine]:
        return [line for line in self.lines if line.is_fulfillable]

    @property
    def unfulfillable_lines(self) -> list[FulfillmentLine]:
        return [line for line in self.lines if not line.is_fulfillable]


# -----------------------------------------------------------------------------
# Small utilities
# -----------------------------------------------------------------------------
def to_builtin(value: Any) -> Any:
    """Recursively convert numpy / pandas scalars into plain Python types (JSON-safe)."""
    if isinstance(value, dict):
        return {str(k): to_builtin(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_builtin(v) for v in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return None if np.isnan(value) else float(value)
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


def to_json(value: Any) -> str:
    """Serialise a tool result for the LLM."""
    return json.dumps(to_builtin(value), indent=1, default=str)


def format_money(amount: float) -> str:
    """Format a currency amount the way it must appear to customers, e.g. $1,234.50."""
    return f"${amount:,.2f}"


def format_unit_price(price: float) -> str:
    """Format a per-unit price with up to 4 decimals (paper is priced per sheet)."""
    text_value = f"{price:.4f}".rstrip("0")
    if len(text_value.split(".")[1]) < 2:
        text_value = f"{price:.2f}"
    return f"${text_value}"


def add_days(iso_date: str, days: int) -> str:
    """Return `iso_date` shifted by `days` days, in YYYY-MM-DD format."""
    return (datetime.fromisoformat(iso_date[:10]) + timedelta(days=days)).strftime("%Y-%m-%d")


def extract_request_date(request_text: str) -> str:
    """Pull the '(Date of request: YYYY-MM-DD)' stamp added by the test harness."""
    match = re.search(r"Date of request:\s*(\d{4}-\d{2}-\d{2})", request_text)
    return match.group(1) if match else datetime.now().strftime("%Y-%m-%d")


def current_stock(item_name: str, as_of_date: str) -> int:
    """Net on-hand units of an item, via the starter helper get_stock_level."""
    stock_df = get_stock_level(item_name, as_of_date)
    if stock_df.empty or pd.isna(stock_df["current_stock"].iloc[0]):
        return 0
    return int(stock_df["current_stock"].iloc[0])


def load_stocked_items() -> dict[str, dict]:
    """Items in the company's regular inventory range with their minimum stock levels."""
    inventory_df = pd.read_sql("SELECT item_name, unit_price, min_stock_level FROM inventory", db_engine)
    return {row["item_name"]: to_builtin(row.to_dict()) for _, row in inventory_df.iterrows()}


def resolve_catalog_item(candidate: Optional[str]) -> Optional[str]:
    """Map an LLM-proposed item name onto an exact catalog name (or None)."""
    if not candidate:
        return None
    exact = CATALOG_BY_LOWER_NAME.get(candidate.strip().lower())
    if exact:
        return exact
    close = difflib.get_close_matches(candidate.strip().lower(), list(CATALOG_BY_LOWER_NAME), n=1, cutoff=0.85)
    return CATALOG_BY_LOWER_NAME[close[0]] if close else None


def pricing_unit_for(catalog_item: Optional[str]) -> str:
    """Paper is priced per sheet, everything else per single unit."""
    if catalog_item and CATALOG[catalog_item]["category"] in SHEET_PRICED_CATEGORIES:
        return "sheet"
    return "unit"


def convert_to_catalog_units(item: RequestedItem) -> tuple[int, str, int]:
    """Convert a customer quantity into pricing units (sheets / single units).

    Returns:
        (quantity_in_pricing_units, package_unit or "", units_per_package)
    """
    unit = item.unit.strip().lower()
    for keyword, default_size in PACKAGE_UNITS:
        if keyword in unit:
            per_package = item.units_per_package or default_size
            if per_package > 1:
                return item.quantity * per_package, keyword, per_package
            break
    return item.quantity, "", 1


# -----------------------------------------------------------------------------
# Domain services (deterministic business logic used by the agents' tools)
# -----------------------------------------------------------------------------
def build_order_lines(items: list[RequestedItem]) -> list[FulfillmentLine]:
    """Normalise the orchestrator's extracted items into order lines."""
    stocked_items = load_stocked_items()
    lines = []
    for line_id, item in enumerate(items, start=1):
        catalog_item = resolve_catalog_item(item.catalog_item)
        quantity, package_unit, units_per_package = convert_to_catalog_units(item)
        lines.append(FulfillmentLine(
            line_id=line_id,
            customer_description=item.customer_description,
            catalog_item=catalog_item,
            quantity=quantity,
            pricing_unit=pricing_unit_for(catalog_item),
            requested_quantity=item.quantity,
            package_unit=package_unit,
            units_per_package=units_per_package,
            unit_cost=CATALOG[catalog_item]["unit_price"] if catalog_item else 0.0,
            is_stocked_item=catalog_item in stocked_items,
        ))
    return lines


def assess_fulfillment(order: OrderContext) -> None:
    """Decide, line by line, whether and how each item can be delivered.

    For each line the assessment checks catalog coverage, on-hand stock
    (get_stock_level), supplier lead time for any shortfall
    (get_supplier_delivery_date) against the customer's deadline, and whether
    the restocking spend fits the procurement budget (get_cash_balance).
    """
    cash = get_cash_balance(order.request_date)
    procurement_budget = max(0.0, min(cash - CASH_RESERVE, cash * MAX_ORDER_PROCUREMENT_SHARE))
    committed_spend = 0.0
    allocated: dict[str, int] = {}  # stock already promised to earlier lines of this order
    in_stock_delivery = add_days(order.request_date, IN_STOCK_DISPATCH_DAYS)

    for line in order.lines:
        if line.catalog_item is None:
            line.status = STATUS_NOT_CARRIED
            line.customer_reason = "This item is not part of our product range, so we are unable to supply it."
            continue

        stock = current_stock(line.catalog_item, order.request_date)
        line.stock_on_hand = stock
        available = max(0, stock - allocated.get(line.catalog_item, 0))

        if available >= line.quantity:
            line.shortfall = 0
            line.estimated_delivery = in_stock_delivery
        else:
            line.shortfall = line.quantity - available
            line.supplier_eta = get_supplier_delivery_date(order.request_date, line.shortfall)
            line.estimated_delivery = add_days(line.supplier_eta, IN_STOCK_DISPATCH_DAYS)

        if order.delivery_deadline and line.estimated_delivery > order.delivery_deadline:
            line.status = STATUS_MISSES_DEADLINE
            partial_hint = ""
            if 0 < available < line.quantity and in_stock_delivery <= order.delivery_deadline:
                partial_hint = (f" We could deliver {available:,} {pluralize(line.pricing_unit, available)} "
                                f"by {order.delivery_deadline} if a reduced quantity would help.")
            line.customer_reason = (
                f"The full quantity would only reach you by {line.estimated_delivery}, "
                f"after your requested delivery date of {order.delivery_deadline}.{partial_hint}"
            )
            continue

        restock_cost = line.shortfall * line.unit_cost
        if restock_cost > 0 and committed_spend + restock_cost > procurement_budget:
            line.status = STATUS_OVER_CAPACITY
            line.customer_reason = (
                "This volume exceeds what we can source for a single order at the moment. "
                "We would be happy to arrange it as a split or scheduled delivery."
            )
            continue

        committed_spend += restock_cost
        allocated[line.catalog_item] = allocated.get(line.catalog_item, 0) + line.quantity
        line.status = STATUS_BACKORDER if line.shortfall else STATUS_IN_STOCK
        line.customer_reason = (
            f"Sourced from our supplier, delivery by {line.estimated_delivery}."
            if line.shortfall
            else f"In stock, delivery by {line.estimated_delivery}."
        )

    order.assessment_done = True


def bulk_discount_rate(total_units: int) -> float:
    """Return the bulk discount rate for an order of `total_units` units."""
    for minimum_units, rate in BULK_DISCOUNT_TIERS:
        if total_units >= minimum_units:
            return rate
    return 0.0


def build_quote(order: OrderContext) -> Quote:
    """Price all fulfillable lines: list price, bulk discount, friendly rounding."""
    fulfillable = order.fulfillable_lines
    total_units = sum(line.quantity for line in fulfillable)
    discount_rate = bulk_discount_rate(total_units)

    quote_lines = []
    for line in fulfillable:
        list_unit_price = round(line.unit_cost * LIST_PRICE_MARKUP, 4)
        quote_lines.append(QuoteLine(
            catalog_item=line.catalog_item,
            customer_description=line.customer_description,
            quantity=line.quantity,
            pricing_unit=line.pricing_unit,
            requested_quantity=line.requested_quantity,
            package_unit=line.package_unit,
            units_per_package=line.units_per_package,
            list_unit_price=list_unit_price,
            line_subtotal=round(list_unit_price * line.quantity, 2),
            line_total=0.0,
            unit_cost=line.unit_cost,
        ))

    subtotal = round(sum(q.line_subtotal for q in quote_lines), 2)
    discount_amount = round(subtotal * discount_rate, 2)
    discounted_total = round(subtotal - discount_amount, 2)
    total = math.floor(discounted_total) if discounted_total >= FRIENDLY_ROUNDING_THRESHOLD else discounted_total

    # Guardrail: never sell below the minimum margin, whatever the discount.
    goods_cost = round(sum(q.unit_cost * q.quantity for q in quote_lines), 2)
    total = round(max(total, goods_cost * MINIMUM_MARGIN_MULTIPLIER), 2)
    rounding_adjustment = round(discounted_total - total, 2)

    # Spread the final total across lines so each sales transaction carries its share.
    allocated_total = 0.0
    for index, quote_line in enumerate(quote_lines):
        if index == len(quote_lines) - 1:
            quote_line.line_total = round(total - allocated_total, 2)
        else:
            share = quote_line.line_subtotal / subtotal if subtotal else 0
            quote_line.line_total = round(total * share, 2)
            allocated_total += quote_line.line_total

    procurement_cost = round(sum(line.shortfall * line.unit_cost for line in fulfillable), 2)
    return Quote(
        lines=quote_lines,
        total_units=total_units,
        subtotal=subtotal,
        discount_rate=discount_rate,
        discount_amount=discount_amount,
        rounding_adjustment=rounding_adjustment,
        total=total,
        procurement_cost=procurement_cost,
        goods_cost=goods_cost,
    )


def discount_calculation(quote: Quote) -> str:
    if not quote.discount_rate:
        return f"no bulk discount (order below {BULK_DISCOUNT_TIERS[-1][0]:,} units)"
    return (f"{quote.discount_rate:.0%} of {format_money(quote.subtotal)} = {format_money(quote.discount_amount)} "
            f"({quote.total_units:,} units in the order)")


def total_calculation(quote: Quote) -> str:
    text_value = format_money(quote.subtotal)
    if quote.discount_amount:
        text_value += f" - {format_money(quote.discount_amount)} bulk discount"
    if quote.rounding_adjustment > 0:
        text_value += f" - {format_money(quote.rounding_adjustment)} courtesy rounding"
    elif quote.rounding_adjustment < 0:
        text_value += f" + {format_money(-quote.rounding_adjustment)} price adjustment"
    return f"{text_value} = {format_money(quote.total)}"


def customer_quote_view(quote: Quote) -> dict:
    """The quote as it may be shown to a customer (no costs or margins)."""
    next_tier = next(
        ((units, rate) for units, rate in reversed(BULK_DISCOUNT_TIERS) if units > quote.total_units),
        None,
    )
    lines = []
    for q in quote.lines:
        line = {
            "item": q.catalog_item,
            "requested_as": q.customer_description,
            "quantity": q.quantity_text,
            "unit_price": q.unit_price_text,
            "line_subtotal": format_money(q.line_subtotal),
            "line_subtotal_calculation": q.subtotal_calculation,
        }
        if q.package_unit:
            line["quantity_conversion"] = q.quantity_note
            line["equivalent_price"] = q.package_price_text
        lines.append(line)
    return {
        "lines": lines,
        "total_units": quote.total_units,
        "subtotal": format_money(quote.subtotal),
        "bulk_discount": f"{quote.discount_rate:.0%}",
        "bulk_discount_amount": format_money(quote.discount_amount),
        "bulk_discount_calculation": discount_calculation(quote),
        "rounding_courtesy_discount": format_money(quote.rounding_adjustment),
        "total": format_money(quote.total),
        "total_calculation": total_calculation(quote),
        "discount_policy": "5% from 500 units, 10% from 2,000 units, 15% from 10,000 units (whole order).",
        "next_discount_tier": (
            f"{next_tier[1]:.0%} off from {next_tier[0]:,} units" if next_tier else "highest tier reached"
        ),
    }


def customer_assessment_view(order: OrderContext) -> list[dict]:
    """Per-line availability as it may be shown to a customer (no stock counts or costs)."""
    return [
        {
            "line": line.line_id,
            "requested_as": line.customer_description,
            "matched_item": line.catalog_item or "no matching product",
            "quantity": line.quantity_text,
            "quantity_conversion": line.quantity_note,
            "status": CUSTOMER_STATUS_LABELS.get(line.status, line.status),
            "estimated_delivery": line.estimated_delivery if line.is_fulfillable else None,
            "explanation": line.customer_reason,
        }
        for line in order.lines
    ]


def financial_health_check(order: OrderContext) -> dict:
    """Project the company's position after this order using generate_financial_report."""
    report = generate_financial_report(order.request_date)
    quote = order.quote
    cash_now = float(report["cash_balance"])
    cash_after = cash_now - quote.procurement_cost + quote.total
    lowest_cash_point = cash_now - quote.procurement_cost  # supplier is paid before the customer
    return {
        "as_of_date": order.request_date,
        "cash_balance": round(cash_now, 2),
        "inventory_value": round(float(report["inventory_value"]), 2),
        "total_assets": round(float(report["total_assets"]), 2),
        "order_revenue": quote.total,
        "procurement_cost_for_order": quote.procurement_cost,
        "projected_cash_after_order": round(cash_after, 2),
        "lowest_cash_point_during_order": round(lowest_cash_point, 2),
        "cash_reserve_required": CASH_RESERVE,
        "meets_cash_reserve": lowest_cash_point >= CASH_RESERVE,
        "sells_above_cost": quote.total >= quote.goods_cost,
        "top_selling_products": report["top_selling_products"],
    }


def book_sale(order: OrderContext) -> SaleRecord:
    """Record one 'sales' transaction per quoted line via create_transaction."""
    quote = order.quote
    transaction_ids = []
    for quote_line in quote.lines:
        transaction_ids.append(create_transaction(
            item_name=quote_line.catalog_item,
            transaction_type="sales",
            quantity=quote_line.quantity,
            price=quote_line.line_total,
            date=order.request_date,
        ))
    delivery_date = max(line.estimated_delivery for line in order.fulfillable_lines)
    order_reference = f"BCP-{order.request_date.replace('-', '')}-{transaction_ids[0]:04d}"
    return SaleRecord(
        order_reference=order_reference,
        total_charged=quote.total,
        transaction_ids=transaction_ids,
        delivery_date=delivery_date,
    )


def procure_backorders(order: OrderContext) -> list[dict]:
    """Buy the shortfall of every back-ordered line from the supplier ('stock_orders')."""
    purchases = []
    for line in order.fulfillable_lines:
        if line.shortfall <= 0 or line.procured:
            continue
        cost = round(line.shortfall * line.unit_cost, 2)
        transaction_id = create_transaction(
            item_name=line.catalog_item,
            transaction_type="stock_orders",
            quantity=line.shortfall,
            price=cost,
            date=order.request_date,
        )
        line.procured = True
        purchase = {"item": line.catalog_item, "units": line.shortfall, "cost": cost,
                    "supplier_eta": line.supplier_eta, "transaction_id": transaction_id}
        purchases.append(purchase)
        order.procurement_log.append(purchase)
    return purchases


def replenish_low_stock(order: OrderContext) -> list[dict]:
    """Top up regular inventory items that fell below their minimum stock level.

    Uses get_all_inventory for the current snapshot (items with zero stock are
    absent from it and therefore treated as 0) and the inventory table for the
    minimum levels. Replenishment is skipped if it would breach the cash reserve.
    """
    if order.replenishment_done:
        return order.replenishment_log
    snapshot = get_all_inventory(order.request_date)
    cash_available = get_cash_balance(order.request_date) - CASH_RESERVE
    purchases = []
    for item_name, item in load_stocked_items().items():
        on_hand = int(snapshot.get(item_name, 0))
        minimum = int(item["min_stock_level"])
        if on_hand >= minimum:
            continue
        reorder_units = minimum * REORDER_UP_TO_MULTIPLIER - on_hand
        cost = round(reorder_units * float(item["unit_price"]), 2)
        if cost > cash_available:
            purchases.append({"item": item_name, "units": 0, "skipped": "insufficient cash above reserve"})
            continue
        transaction_id = create_transaction(item_name, "stock_orders", reorder_units, cost, order.request_date)
        cash_available -= cost
        purchases.append({
            "item": item_name, "on_hand_before": on_hand, "min_stock_level": minimum,
            "units": reorder_units, "cost": cost,
            "supplier_eta": get_supplier_delivery_date(order.request_date, reorder_units),
            "transaction_id": transaction_id,
        })
    order.replenishment_log = purchases
    order.replenishment_done = True
    return purchases


def render_customer_summary(order: OrderContext) -> str:
    """Deterministic customer reply, used as a fallback if the LLM cannot answer."""
    parts = ["Thank you for contacting Beaver's Choice Paper Company."]
    if not order.assessment_done:
        parts.append("We were unable to complete the automatic processing of your request. "
                     "A member of our team will review it and contact you shortly with availability and pricing.")
        return "\n".join(parts)
    quote_is_offered = order.sale is not None or (order.quote and order.customer_intent == "quote_only")
    if order.sale:
        parts.append(f"Your order {order.sale.order_reference} is confirmed.")
    elif quote_is_offered:
        parts.append("Here is your quote.")
    if quote_is_offered:
        for quote_line in order.quote.lines:
            parts.append(f"- {quote_line.catalog_item} (requested as '{quote_line.customer_description}'):")
            if quote_line.package_unit:
                parts.append(f"  Quantity: {quote_line.quantity_note}")
                parts.append(f"  Price: {quote_line.unit_price_text} (equivalent to {quote_line.package_price_text})")
            else:
                parts.append(f"  Quantity: {quote_line.quantity_text}")
                parts.append(f"  Price: {quote_line.unit_price_text}")
            parts.append(f"  Line subtotal: {quote_line.subtotal_calculation}")
        parts.append(f"Subtotal: {format_money(order.quote.subtotal)}")
        parts.append(f"Bulk discount: {discount_calculation(order.quote)}")
        parts.append(f"Total: {total_calculation(order.quote)}")
    if order.sale:
        parts.append(f"Estimated delivery: {order.sale.delivery_date}.")
    for line in order.unfulfillable_lines:
        parts.append(f"- Unable to supply '{line.customer_description}': {line.customer_reason}")
    if not order.sale and not order.quote:
        parts.append("Unfortunately we are unable to fulfil this request at this time. "
                     "A member of our team will contact you to discuss alternatives.")
    return "\n".join(parts)


# -----------------------------------------------------------------------------
# Agent prompts
# -----------------------------------------------------------------------------
CATALOG_LISTING = "\n".join(f"- {item['item_name']} ({item['category']})" for item in paper_supplies)

ORCHESTRATOR_PROMPT = f"""
You are the customer service orchestrator of the Beaver's Choice Paper Company.
You manage a team of specialist agents and you are the only one who talks to the customer.

STANDARD OPERATING PROCEDURE (follow in order, one tool call at a time):
1. Call `record_customer_order` with every item the customer asks for, the delivery deadline
   (YYYY-MM-DD, or null if none is given) and the intent ("order" if they want to buy / place
   an order, "quote_only" if they only ask for a price).
2. Call `delegate_to_inventory_agent` to learn which items can be delivered and when.
3. If at least one item can be fulfilled, call `delegate_to_quoting_agent` for pricing.
4. If the intent is "order" and a quote exists, call `delegate_to_sales_agent` to finalise the sale.
5. Write the final reply to the customer.

ITEM EXTRACTION RULES for `record_customer_order`:
- Copy the quantity and unit exactly as stated (e.g. 500 reams -> quantity=500, unit="ream").
  Unit conversion is done by the system. Only fill units_per_package when the customer states it.
- Map each item to the single closest catalog item, using the exact catalog name:
  * The main product noun decides the item; colour or quality adjectives (colorful, assorted
    colors, white, high-quality, sturdy) never change it: "colorful poster paper" -> "Poster paper",
    "cardstock in various colors" -> "Cardstock", "colorful construction paper" -> "Construction paper".
    Use "Colored paper" only when the product itself is just colored paper.
  * "A4 paper", "printer paper", "printing paper", "copy paper", "white paper" -> the plain paper
    the customer describes (e.g. "A4 paper" or "Standard copy paper").
  * Descriptive words like glossy, matte, recycled, colored, cardstock, construction, kraft,
    poster, heavyweight decide the item (e.g. "A4 glossy paper" -> "Glossy paper",
    "heavy cardstock" -> "Cardstock", "poster board 24x36" -> "Large poster paper (24x36 inches)",
    "streamers" -> "Party streamers", "table napkins" -> "Paper napkins", "washi tape" ->
    "Decorative adhesive tape (washi tape)").
  * We only stock standard A4/letter formats. Items in other sheet sizes (A3, A5, ...) and
    products that are not in the catalog (balloons, tickets, cardboard, ...) get catalog_item=null.
    Never substitute a different size or product.

CATALOG:
{CATALOG_LISTING}

FINAL REPLY RULES:
- Address the customer politely; summarise every requested item and what happens to it.
- For confirmed orders and quotes include: the order reference (if confirmed), and for each item its
  quantity, its unit price WITH the price basis exactly as given (e.g. "$0.05 per sheet"), and the
  line subtotal calculation exactly as given (e.g. "250,000 sheets x $0.05 per sheet = $12,500.00").
- If an item was requested in reams, boxes or packs, also show the `quantity_conversion` exactly as
  given (e.g. "500 reams x 500 sheets per ream = 250,000 sheets") and the `equivalent_price`
  (e.g. "$25.00 per ream"), so the customer can verify the line subtotal.
- Then show the subtotal, the bulk discount calculation and why it applies, the total calculation
  and the total exactly as given by the quote (e.g. "$1,234.00"), and the estimated delivery date.
- For items that cannot be supplied, give the reason provided by the inventory agent and, where
  helpful, an alternative (e.g. a reduced quantity or later date).
- Never reveal internal information: costs, margins, markups, cash or financial figures, stock
  counts, supplier details beyond delivery dates, tool names or error messages.
- Do not invent prices, dates or items. Use only data returned by your team.
- Sign off as "The Beaver's Choice Paper Company Sales Team". Never use placeholders such as
  [Your Name] or [Customer Name].
"""

INVENTORY_AGENT_PROMPT = f"""
You are the Inventory Agent of the Beaver's Choice Paper Company. You own stock levels,
supplier lead times and purchasing decisions (stock orders). You never talk to customers.

When asked to ASSESS an order:
- Call `assess_order_fulfillment` (it checks stock, supplier lead times against the customer's
  deadline, and the procurement budget for every line). You may call `check_stock_level`,
  `get_inventory_snapshot` or `estimate_supplier_delivery` for extra detail if useful.
- Reply with a short internal summary: which lines are in stock, which need restocking and
  which cannot be fulfilled, with the reasons.

When asked to PROCURE after a confirmed sale:
- Call `procure_backordered_items` to buy the shortfall of back-ordered lines.
- Then call `replenish_low_stock_items` so regular items do not stay below their minimum level
  (reorder point policy: reorder up to {REORDER_UP_TO_MULTIPLIER}x the minimum stock level).
- Reply with a short internal summary of the purchases made.
"""

QUOTING_AGENT_PROMPT = """
You are the Quoting Agent of the Beaver's Choice Paper Company. You produce competitive,
transparent quotes that encourage larger orders.

Steps:
1. Call `find_similar_past_quotes` with 1-3 short keywords from the order (e.g. a product
   word such as "cardstock" or the event type such as "ceremony") to see how comparable
   orders were priced and explained in the past.
2. Call `calculate_quote` to price the fulfillable items. The pricing policy (list prices,
   bulk discount tiers, courtesy rounding) is applied by the tool - never change its numbers.
3. Reply with a 2-4 sentence customer-friendly explanation of the price: mention the bulk
   discount and why it applies (or how many more units would unlock the next tier), the
   courtesy rounding, and how it compares to similar past orders if relevant. Paper is priced
   per sheet: when an item was ordered in reams/boxes/packs, say so and quote the conversion
   and equivalent package price returned by the tool.
   Never mention costs, margins, markups or internal policies beyond the discount tiers.
"""

SALES_AGENT_PROMPT = f"""
You are the Sales Agent of the Beaver's Choice Paper Company. You finalise sales and
protect the company's financial health. You never talk to customers directly.

Steps:
1. Call `check_cash_balance`.
2. If the quote total exceeds ${LARGE_ORDER_THRESHOLD:,.0f} or the order needs restocking from the supplier,
   call `run_financial_health_check` before deciding.
3. If the order keeps cash above the required reserve and sells above cost, call
   `finalize_sale`. Otherwise call `decline_sale` with a short, customer-safe reason.
4. Reply with a one-sentence internal summary of your decision (include the order reference).
"""

BUSINESS_ADVISOR_PROMPT = """
You are the Business Advisor of the Beaver's Choice Paper Company. You analyse the
transactions handled by the multi-agent system and proactively recommend changes to
business operations that improve efficiency and revenue.

Use `get_financial_report`, `get_order_outcomes` and `get_inventory_snapshot_for_advisor`,
then write a concise markdown report with these sections:
## Performance summary   (revenue, fulfilment rate, cash and inventory position)
## Lost demand           (what customers asked for that we could not supply, and why)
## Recommendations       (3-5 specific, actionable recommendations, each with the evidence)
Base every statement on the data returned by your tools. Use the pre-computed KPIs in the
`summary` of get_order_outcomes for revenue, profit and fulfilment figures - do not estimate them.
"""


# -----------------------------------------------------------------------------
# Agent definitions
# -----------------------------------------------------------------------------
inventory_agent = Agent(
    llm_model,
    deps_type=OrderContext,
    output_type=str,
    name="inventory_agent",
    system_prompt=INVENTORY_AGENT_PROMPT,
    model_settings=AGENT_MODEL_SETTINGS,
    retries=2,
)

quoting_agent = Agent(
    llm_model,
    deps_type=OrderContext,
    output_type=str,
    name="quoting_agent",
    system_prompt=QUOTING_AGENT_PROMPT,
    model_settings=AGENT_MODEL_SETTINGS,
    retries=2,
)

sales_agent = Agent(
    llm_model,
    deps_type=OrderContext,
    output_type=str,
    name="sales_agent",
    system_prompt=SALES_AGENT_PROMPT,
    model_settings=AGENT_MODEL_SETTINGS,
    retries=2,
)

orchestrator_agent = Agent(
    llm_model,
    deps_type=OrderContext,
    output_type=str,
    name="orchestrator_agent",
    system_prompt=ORCHESTRATOR_PROMPT,
    model_settings=AGENT_MODEL_SETTINGS,
    retries=3,
)


@dataclass
class AdvisorContext:
    """Deps for the business advisor: the reporting date and all order outcomes."""

    as_of_date: str
    order_outcomes: list[dict]


business_advisor_agent = Agent(
    llm_model,
    deps_type=AdvisorContext,
    output_type=str,
    name="business_advisor_agent",
    system_prompt=BUSINESS_ADVISOR_PROMPT,
    model_settings={"temperature": 0.3},
    retries=2,
)


# -----------------------------------------------------------------------------
# Inventory Agent tools
# -----------------------------------------------------------------------------
@inventory_agent.tool
def check_stock_level(ctx: RunContext[OrderContext], item_name: str) -> str:
    """Return the on-hand stock of one catalog item as of the request date.

    Args:
        item_name: Exact catalog item name.
    """
    tracer.tool("Inventory Agent", "check_stock_level", item_name)
    resolved = resolve_catalog_item(item_name)
    if resolved is None:
        return to_json({"item_name": item_name, "error": "not a catalog item"})
    return to_json({"item_name": resolved, "current_stock": current_stock(resolved, ctx.deps.request_date)})


@inventory_agent.tool
def get_inventory_snapshot(ctx: RunContext[OrderContext]) -> str:
    """Return every item with positive stock as of the request date."""
    tracer.tool("Inventory Agent", "get_inventory_snapshot", ctx.deps.request_date)
    return to_json(get_all_inventory(ctx.deps.request_date))


@inventory_agent.tool
def estimate_supplier_delivery(ctx: RunContext[OrderContext], quantity: int) -> str:
    """Estimate when the supplier would deliver an order of `quantity` units placed today.

    Args:
        quantity: Number of units to order from the supplier.
    """
    tracer.tool("Inventory Agent", "estimate_supplier_delivery", f"{quantity} units")
    return to_json({
        "order_date": ctx.deps.request_date,
        "quantity": quantity,
        "supplier_delivery_date": get_supplier_delivery_date(ctx.deps.request_date, quantity),
    })


@inventory_agent.tool
def assess_order_fulfillment(ctx: RunContext[OrderContext]) -> str:
    """Assess every order line: stock, restock needs, delivery date vs. deadline and budget."""
    order = ctx.deps
    tracer.tool("Inventory Agent", "assess_order_fulfillment", f"{len(order.lines)} line(s)")
    assess_fulfillment(order)
    return to_json({
        "delivery_deadline": order.delivery_deadline,
        "lines": [
            {
                "line": line.line_id,
                "item": line.catalog_item,
                "quantity": line.quantity,
                "stock_on_hand": line.stock_on_hand,
                "shortfall_to_order": line.shortfall,
                "supplier_eta": line.supplier_eta,
                "estimated_customer_delivery": line.estimated_delivery,
                "status": line.status,
                "reason": line.customer_reason,
            }
            for line in order.lines
        ],
    })


@inventory_agent.tool
def procure_backordered_items(ctx: RunContext[OrderContext]) -> str:
    """Place supplier stock orders for the shortfall of all back-ordered lines of the confirmed sale."""
    order = ctx.deps
    if order.sale is None:
        raise ModelRetry("There is no confirmed sale yet, so nothing may be procured.")
    tracer.tool("Inventory Agent", "procure_backordered_items")
    purchases = procure_backorders(order)
    return to_json({"purchases": purchases} if purchases else {"purchases": [], "note": "no back-ordered lines"})


@inventory_agent.tool
def replenish_low_stock_items(ctx: RunContext[OrderContext]) -> str:
    """Reorder regular inventory items whose stock fell below their minimum level."""
    tracer.tool("Inventory Agent", "replenish_low_stock_items")
    purchases = replenish_low_stock(ctx.deps)
    return to_json({"replenishment_orders": purchases} if purchases else {"replenishment_orders": [], "note": "all items above minimum"})


# -----------------------------------------------------------------------------
# Quoting Agent tools
# -----------------------------------------------------------------------------
@quoting_agent.tool
def find_similar_past_quotes(ctx: RunContext[OrderContext], search_terms: list[str]) -> str:
    """Search the quote history for comparable past orders.

    Args:
        search_terms: 1-3 short keywords (product words or the event type). All terms must match.
    """
    tracer.tool("Quoting Agent", "find_similar_past_quotes", ", ".join(search_terms))
    matches = search_quote_history(search_terms, limit=5)
    if not matches and len(search_terms) > 1:
        # Broaden the search: the helper requires every term to match.
        matches = search_quote_history(search_terms[:1], limit=5)
    ctx.deps.similar_quotes_found = len(matches)
    return to_json([
        {
            "total_amount": match["total_amount"],
            "order_size": match["order_size"],
            "event_type": match["event_type"],
            "quote_explanation": match["quote_explanation"],
        }
        for match in matches
    ])


@quoting_agent.tool
def calculate_quote(ctx: RunContext[OrderContext]) -> str:
    """Price all fulfillable lines with list prices, bulk discount tiers and courtesy rounding."""
    order = ctx.deps
    if not order.fulfillable_lines:
        raise ModelRetry("No line of this order can be fulfilled, so there is nothing to quote.")
    tracer.tool("Quoting Agent", "calculate_quote", f"{len(order.fulfillable_lines)} line(s)")
    order.quote = build_quote(order)
    return to_json(customer_quote_view(order.quote))


# -----------------------------------------------------------------------------
# Sales Agent tools
# -----------------------------------------------------------------------------
@sales_agent.tool
def check_cash_balance(ctx: RunContext[OrderContext]) -> str:
    """Return the company's cash balance as of the request date."""
    tracer.tool("Sales Agent", "check_cash_balance", ctx.deps.request_date)
    return to_json({"as_of_date": ctx.deps.request_date,
                    "cash_balance": round(get_cash_balance(ctx.deps.request_date), 2),
                    "cash_reserve_required": CASH_RESERVE})


@sales_agent.tool
def run_financial_health_check(ctx: RunContext[OrderContext]) -> str:
    """Generate the financial report and project cash after this order (for large orders)."""
    tracer.tool("Sales Agent", "run_financial_health_check", ctx.deps.request_date)
    ctx.deps.health_check_done = True
    return to_json(financial_health_check(ctx.deps))


@sales_agent.tool
def finalize_sale(ctx: RunContext[OrderContext]) -> str:
    """Book the quoted order as sales transactions and return the order reference."""
    order = ctx.deps
    if order.quote is None:
        raise ModelRetry("There is no quote to finalise.")
    if order.sale is not None:
        return to_json({"order_reference": order.sale.order_reference, "note": "already finalised"})
    needs_check = order.quote.total > LARGE_ORDER_THRESHOLD or order.quote.procurement_cost > 0
    if needs_check and not order.health_check_done:
        raise ModelRetry("This order is large or needs restocking: call run_financial_health_check first.")
    if get_cash_balance(order.request_date) - order.quote.procurement_cost < CASH_RESERVE:
        raise ModelRetry("Finalising would breach the cash reserve. Call decline_sale instead.")
    tracer.tool("Sales Agent", "finalize_sale", format_money(order.quote.total))
    order.sale = book_sale(order)
    return to_json({
        "order_reference": order.sale.order_reference,
        "total_charged": format_money(order.sale.total_charged),
        "estimated_delivery": order.sale.delivery_date,
        "sales_transactions": len(order.sale.transaction_ids),
    })


@sales_agent.tool
def decline_sale(ctx: RunContext[OrderContext], customer_safe_reason: str) -> str:
    """Decline the order. The reason will be shown to the customer, so keep it free of internal figures.

    Args:
        customer_safe_reason: Short explanation suitable for the customer.
    """
    tracer.tool("Sales Agent", "decline_sale", customer_safe_reason)
    ctx.deps.decline_reason = customer_safe_reason
    return "Order declined."


# -----------------------------------------------------------------------------
# Orchestrator tools (delegation to the worker agents)
# -----------------------------------------------------------------------------
@orchestrator_agent.tool
def record_customer_order(
    ctx: RunContext[OrderContext],
    items: list[RequestedItem],
    delivery_deadline: Optional[str],
    customer_intent: Literal["order", "quote_only"],
) -> str:
    """Record the items, deadline and intent extracted from the customer's request.

    Args:
        items: Every item the customer asks for, mapped to the catalog where possible.
        delivery_deadline: Requested delivery date as YYYY-MM-DD, or null if none was given.
        customer_intent: "order" to buy now, "quote_only" if the customer only wants a price.
    """
    order = ctx.deps
    if order.assessment_done:
        raise ModelRetry("The order has already been assessed; continue with the next step.")
    if delivery_deadline:
        try:
            delivery_deadline = datetime.fromisoformat(delivery_deadline.strip()[:10]).strftime("%Y-%m-%d")
        except ValueError:
            raise ModelRetry("delivery_deadline must be a date in YYYY-MM-DD format or null.")
    if not items:
        raise ModelRetry("Record at least one requested item.")

    order.delivery_deadline = delivery_deadline
    order.customer_intent = customer_intent
    order.lines = build_order_lines(items)
    order.items_recorded = True
    tracer.tool("Orchestrator", "record_customer_order",
                f"{len(order.lines)} item(s), deadline {delivery_deadline}, intent {customer_intent}")
    return to_json({
        "request_date": order.request_date,
        "delivery_deadline": order.delivery_deadline,
        "lines": [
            {"line": line.line_id, "requested_as": line.customer_description,
             "matched_item": line.catalog_item, "quantity_in_catalog_units": line.quantity,
             "quantity_note": line.quantity_note}
            for line in order.lines
        ],
    })


@orchestrator_agent.tool
async def delegate_to_inventory_agent(ctx: RunContext[OrderContext]) -> str:
    """Ask the Inventory Agent which items can be delivered by the deadline, and when."""
    order = ctx.deps
    if not order.items_recorded:
        raise ModelRetry("Call record_customer_order first.")
    tracer.handoff("Orchestrator", "Inventory Agent", "check availability & lead times")
    result = await inventory_agent.run(
        f"ASSESS the order placed on {order.request_date} "
        f"(deadline: {order.delivery_deadline or 'none'}) with {len(order.lines)} line(s).",
        deps=order,
        usage=ctx.usage,
    )
    if not order.assessment_done:  # guardrail: the assessment is mandatory
        tracer.note("Orchestrator", "inventory assessment missing - running it directly")
        assess_fulfillment(order)
    tracer.note("Inventory Agent", f"{len(order.fulfillable_lines)} of {len(order.lines)} line(s) can be fulfilled")
    return to_json({
        "internal_note_do_not_share": result.output,
        "lines": customer_assessment_view(order),
        "fulfillable_lines": len(order.fulfillable_lines),
    })


@orchestrator_agent.tool
async def delegate_to_quoting_agent(ctx: RunContext[OrderContext]) -> str:
    """Ask the Quoting Agent to price the fulfillable items and explain the price."""
    order = ctx.deps
    if not order.assessment_done:
        raise ModelRetry("Call delegate_to_inventory_agent first.")
    if not order.fulfillable_lines:
        return "No items can be fulfilled, so there is nothing to quote. Explain the reasons to the customer."
    if order.sale is not None:  # the booked price is final; never re-quote a confirmed sale
        return to_json({"quote": customer_quote_view(order.quote), "pricing_explanation": order.quote_explanation})
    tracer.handoff("Orchestrator", "Quoting Agent", "price the order")
    lines_text = "\n".join(
        f"- {line.quantity_text} of {line.catalog_item} (requested as '{line.customer_description}'"
        + (f"; {line.quantity_note}" if line.package_unit else "") + ")"
        for line in order.fulfillable_lines
    )
    result = await quoting_agent.run(
        f"Prepare a quote for this order from the customer request:\n\"{order.request_text}\"\n"
        f"Fulfillable items:\n{lines_text}",
        deps=order,
        usage=ctx.usage,
    )
    if order.quote is None:  # guardrail: the quote must come from the pricing tool
        order.quote = build_quote(order)
    order.quote_explanation = result.output
    tracer.note("Quoting Agent", f"quote total {format_money(order.quote.total)} "
                                 f"({order.quote.discount_rate:.0%} bulk discount)")
    return to_json({"quote": customer_quote_view(order.quote), "pricing_explanation": order.quote_explanation})


@orchestrator_agent.tool
async def delegate_to_sales_agent(ctx: RunContext[OrderContext]) -> str:
    """Ask the Sales Agent to finalise the quoted order. Restocking is then handed to the Inventory Agent."""
    order = ctx.deps
    if order.quote is None:
        raise ModelRetry("Call delegate_to_quoting_agent first.")
    if order.customer_intent != "order":
        return "The customer only asked for a quote, so no sale is booked."
    if order.sale is None and order.decline_reason is None:
        tracer.handoff("Orchestrator", "Sales Agent", f"finalise sale of {format_money(order.quote.total)}")
        sales_result = await sales_agent.run(
            f"Finalise the order dated {order.request_date}. Quote total {format_money(order.quote.total)}, "
            f"{sum(1 for line in order.fulfillable_lines if line.shortfall)} line(s) need restocking.",
            deps=order,
            usage=ctx.usage,
        )
        tracer.note("Sales Agent", sales_result.output)

    if order.sale is None:
        return to_json({"status": "declined", "reason": order.decline_reason or "The order could not be finalised."})

    # Hand-off: after the sale the Inventory Agent buys back-ordered stock and replenishes.
    if not order.replenishment_done:
        tracer.handoff("Sales Agent", "Inventory Agent", "procure back-orders & replenish stock")
        procurement_result = await inventory_agent.run(
            f"PROCURE for confirmed sale {order.sale.order_reference}.",
            deps=order,
            usage=ctx.usage,
        )
        tracer.note("Inventory Agent", procurement_result.output)
        # Guardrails: back-orders must always be bought; replenishment always checked.
        if any(line.shortfall and not line.procured for line in order.fulfillable_lines):
            tracer.note("Orchestrator", "back-orders not procured by agent - procuring directly")
            procure_backorders(order)
        if not order.replenishment_done:
            replenish_low_stock(order)

    return to_json({
        "status": "confirmed",
        "order_reference": order.sale.order_reference,
        "total_charged": format_money(order.sale.total_charged),
        "estimated_delivery": order.sale.delivery_date,
        "line_deliveries": [
            {"item": line.catalog_item, "quantity": line.quantity, "estimated_delivery": line.estimated_delivery}
            for line in order.fulfillable_lines
        ],
    })


@orchestrator_agent.output_validator
def validate_customer_reply(ctx: RunContext[OrderContext], reply: str) -> str:
    """Enforce the standard operating procedure and protect sensitive information."""
    order = ctx.deps
    if not order.items_recorded:
        raise ModelRetry("First call record_customer_order with the customer's items.")
    if not order.assessment_done:
        raise ModelRetry("Call delegate_to_inventory_agent before replying.")
    if order.fulfillable_lines and order.quote is None:
        raise ModelRetry("Some items can be fulfilled: call delegate_to_quoting_agent before replying.")
    if (order.customer_intent == "order" and order.quote is not None
            and order.sale is None and order.decline_reason is None):
        raise ModelRetry("The customer placed an order: call delegate_to_sales_agent before replying.")

    lowered = reply.lower()
    leaked_terms = [term for term in FORBIDDEN_CUSTOMER_TERMS if term in lowered]
    if leaked_terms:
        raise ModelRetry(f"Rewrite the reply without internal information ({', '.join(leaked_terms)}).")
    placeholder = re.search(r"\[[^\]]*(name|company|title)[^\]]*\]", reply, flags=re.IGNORECASE)
    if placeholder:
        raise ModelRetry(f"Remove the placeholder {placeholder.group(0)}; sign as the Sales Team.")
    if order.sale and order.sale.order_reference not in reply:
        raise ModelRetry(f"Include the order reference {order.sale.order_reference} in the reply.")
    quote_is_offered = order.quote is not None and (order.sale is not None or order.customer_intent == "quote_only")
    if not quote_is_offered:
        return reply

    missing = []
    for q in order.quote.lines:
        required = [q.unit_price_text, format_money(q.line_subtotal)]
        if q.package_unit:
            required += [q.per_package_text, q.quantity_text, q.package_price_text]
        missing += [f"{q.catalog_item}: '{phrase}'" for phrase in required if phrase not in reply]
    if missing:
        raise ModelRetry(
            "For every quoted item show the quantity conversion (for reams/boxes/packs), the unit price "
            "with its basis and the line subtotal calculation exactly as given by the quote. Missing: "
            + "; ".join(missing))
    if order.quote.discount_amount and format_money(order.quote.discount_amount) not in reply:
        raise ModelRetry(f"Show the bulk discount amount {format_money(order.quote.discount_amount)}.")
    if format_money(order.quote.total) not in reply:
        raise ModelRetry(f"State the total exactly as {format_money(order.quote.total)}.")
    return reply


# -----------------------------------------------------------------------------
# Business Advisor tools
# -----------------------------------------------------------------------------
@business_advisor_agent.tool
def get_financial_report(ctx: RunContext[AdvisorContext]) -> str:
    """Return the company's financial report (cash, inventory value, top sellers)."""
    tracer.tool("Business Advisor", "get_financial_report", ctx.deps.as_of_date)
    report = generate_financial_report(ctx.deps.as_of_date)
    low_stock = [item for item in report["inventory_summary"] if item["stock"] <= 0]
    report["inventory_summary"] = [item for item in report["inventory_summary"] if item["stock"] > 0]
    report["items_out_of_stock"] = [item["item_name"] for item in low_stock]
    # The starter seeds opening cash as a 'sales' row without an item; it is not product revenue.
    report["top_selling_products"] = [p for p in report["top_selling_products"] if p.get("item_name")]
    report["opening_cash_balance"] = round(get_cash_balance(datetime(2025, 1, 1).isoformat()), 2)
    return to_json(report)


def summarize_order_outcomes(order_outcomes: list[dict]) -> dict:
    """Aggregate KPIs over all handled requests (fulfilment rate, revenue, lost demand)."""
    outcome_counts: dict[str, int] = {}
    for outcome in order_outcomes:
        label = outcome["outcome"].split(" (")[0]
        outcome_counts[label] = outcome_counts.get(label, 0) + 1
    all_lines = [line for outcome in order_outcomes for line in outcome["lines"]]
    lost_lines = [line for line in all_lines if line["status"] not in FULFILLABLE_STATUSES]
    lost_by_reason: dict[str, int] = {}
    lost_by_item: dict[str, int] = {}
    for line in lost_lines:
        lost_by_reason[line["status"]] = lost_by_reason.get(line["status"], 0) + 1
        key = line["item"] or f"not carried: {line['requested_as']}"
        lost_by_item[key] = lost_by_item.get(key, 0) + line["quantity"]
    revenue = round(sum(o["revenue"] for o in order_outcomes), 2)
    goods_cost = round(sum(o["goods_cost"] for o in order_outcomes), 2)
    return {
        "requests_handled": len(order_outcomes),
        "requests_by_outcome": outcome_counts,
        "order_lines_requested": len(all_lines),
        "order_lines_fulfilled": len(all_lines) - len(lost_lines),
        "line_fill_rate": round((len(all_lines) - len(lost_lines)) / len(all_lines), 3) if all_lines else None,
        "sales_revenue": revenue,
        "cost_of_goods_sold": goods_cost,
        "gross_profit": round(revenue - goods_cost, 2),
        "procurement_spend_for_backorders": round(sum(o["procurement_spend"] for o in order_outcomes), 2),
        "replenishment_spend": round(sum(o["replenishment_spend"] for o in order_outcomes), 2),
        "lost_lines_by_reason": lost_by_reason,
        "lost_units_by_item": dict(sorted(lost_by_item.items(), key=lambda kv: -kv[1])),
    }


@business_advisor_agent.tool
def get_order_outcomes(ctx: RunContext[AdvisorContext]) -> str:
    """Return KPIs over all requests plus the outcome of every individual request."""
    tracer.tool("Business Advisor", "get_order_outcomes", f"{len(ctx.deps.order_outcomes)} request(s)")
    return to_json({"summary": summarize_order_outcomes(ctx.deps.order_outcomes),
                    "requests": ctx.deps.order_outcomes})


@business_advisor_agent.tool
def get_inventory_snapshot_for_advisor(ctx: RunContext[AdvisorContext]) -> str:
    """Return the current stock of every item with positive inventory."""
    tracer.tool("Business Advisor", "get_inventory_snapshot", ctx.deps.as_of_date)
    return to_json(get_all_inventory(ctx.deps.as_of_date))


# -----------------------------------------------------------------------------
# Multi-agent system facade
# -----------------------------------------------------------------------------
class BeaversChoiceMultiAgentSystem:
    """Entry point that runs the orchestrator for each request and keeps an outcome log."""

    def __init__(self) -> None:
        self.order_outcomes: list[dict] = []
        if not OPENAI_API_KEY:
            print("WARNING: UDACITY_OPENAI_API_KEY is not set - LLM calls will fail. "
                  "Add it to a .env file next to this script.")

    def handle_request(self, request_with_date: str) -> str:
        """Process one customer request end-to-end and return the reply to the customer."""
        request_id = len(self.order_outcomes) + 1
        order = OrderContext(
            request_id=request_id,
            request_text=request_with_date,
            request_date=extract_request_date(request_with_date),
        )
        tracer.handoff("Customer", "Orchestrator", f"request #{request_id} received ({order.request_date})")
        reply = None
        for attempt in range(1, MAX_REQUEST_ATTEMPTS + 1):
            try:
                result = orchestrator_agent.run_sync(request_with_date, deps=order)
                reply = result.output.strip()
                break
            except Exception as error:  # never expose internal errors to the customer
                print(f"ERROR while processing request #{request_id} (attempt {attempt}): "
                      f"{type(error).__name__}: {str(error)[:160]}")
                ledger_touched = order.sale is not None or order.procurement_log or order.replenishment_log
                if ledger_touched or attempt == MAX_REQUEST_ATTEMPTS:
                    break
                # Nothing was written to the ledger yet, so the request can safely start over.
                time.sleep(RETRY_BACKOFF_SECONDS * attempt)
                order = OrderContext(request_id=request_id, request_text=request_with_date,
                                     request_date=order.request_date)
        if reply is None:
            reply = render_customer_summary(order)
        tracer.handoff("Orchestrator", "Customer", self._status_label(order))
        self.order_outcomes.append(self._outcome_record(order))
        return reply

    @staticmethod
    def _status_label(order: OrderContext) -> str:
        if order.sale and order.unfulfillable_lines:
            return f"partially fulfilled ({order.sale.order_reference})"
        if order.sale:
            return f"fulfilled ({order.sale.order_reference})"
        if order.quote and order.customer_intent == "quote_only":
            return "quote provided"
        return "not fulfilled"

    def _outcome_record(self, order: OrderContext) -> dict:
        """Internal record of one request, used by the Business Advisor."""
        return {
            "request_id": order.request_id,
            "request_date": order.request_date,
            "delivery_deadline": order.delivery_deadline,
            "outcome": self._status_label(order),
            "revenue": order.sale.total_charged if order.sale else 0.0,
            "goods_cost": order.quote.goods_cost if order.sale else 0.0,
            "discount_rate": order.quote.discount_rate if order.quote else None,
            "procurement_spend": round(sum(p["cost"] for p in order.procurement_log), 2),
            "replenishment_spend": round(sum(p.get("cost", 0) for p in order.replenishment_log), 2),
            "lines": [
                {"requested_as": line.customer_description, "item": line.catalog_item,
                 "quantity": line.quantity, "status": line.status}
                for line in order.lines
            ],
        }

    def run_business_advisor(self, as_of_date: str) -> str:
        """Ask the Business Advisor Agent for recommendations based on all outcomes."""
        tracer.handoff("Orchestrator", "Business Advisor", "analyse all transactions")
        try:
            result = business_advisor_agent.run_sync(
                f"Analyse business performance up to {as_of_date} and recommend improvements.",
                deps=AdvisorContext(as_of_date=as_of_date, order_outcomes=self.order_outcomes),
            )
            return result.output
        except Exception as error:
            print(f"ERROR while running the business advisor: {type(error).__name__}: {error}")
            return "Business advisor report unavailable."


def call_your_multi_agent_system(request_with_date: str, system: Optional[BeaversChoiceMultiAgentSystem] = None) -> str:
    """Convenience wrapper: process a single request with a (new) multi-agent system."""
    return (system or BeaversChoiceMultiAgentSystem()).handle_request(request_with_date)


# Run your test scenarios by writing them here. Make sure to keep track of them.

def run_test_scenarios():

    print("Initializing Database...")
    init_database(db_engine)
    try:
        quote_requests_sample = pd.read_csv("quote_requests_sample.csv")
        quote_requests_sample["request_date"] = pd.to_datetime(
            quote_requests_sample["request_date"], format="%m/%d/%y", errors="coerce"
        )
        quote_requests_sample.dropna(subset=["request_date"], inplace=True)
        quote_requests_sample = quote_requests_sample.sort_values("request_date")
    except Exception as e:
        print(f"FATAL: Error loading test data: {e}")
        return

    # Get initial state
    initial_date = quote_requests_sample["request_date"].min().strftime("%Y-%m-%d")
    report = generate_financial_report(initial_date)
    current_cash = report["cash_balance"]
    current_inventory = report["inventory_value"]

    ############
    ############
    ############
    # INITIALIZE YOUR MULTI AGENT SYSTEM HERE
    ############
    ############
    ############
    multi_agent_system = BeaversChoiceMultiAgentSystem()

    results = []
    for idx, row in quote_requests_sample.iterrows():
        request_date = row["request_date"].strftime("%Y-%m-%d")

        print(f"\n=== Request {idx+1} ===")
        print(f"Context: {row['job']} organizing {row['event']}")
        print(f"Request Date: {request_date}")
        print(f"Cash Balance: ${current_cash:.2f}")
        print(f"Inventory Value: ${current_inventory:.2f}")

        # Process request
        request_with_date = f"{row['request']} (Date of request: {request_date})"

        ############
        ############
        ############
        # USE YOUR MULTI AGENT SYSTEM TO HANDLE THE REQUEST
        ############
        ############
        ############

        response = multi_agent_system.handle_request(request_with_date)

        # Update state
        report = generate_financial_report(request_date)
        current_cash = report["cash_balance"]
        current_inventory = report["inventory_value"]

        print(f"Response: {response}")
        print(f"Updated Cash: ${current_cash:.2f}")
        print(f"Updated Inventory: ${current_inventory:.2f}")

        results.append(
            {
                "request_id": idx + 1,
                "request_date": request_date,
                "cash_balance": current_cash,
                "inventory_value": current_inventory,
                "response": response,
            }
        )

        time.sleep(1)

    # Final report
    final_date = quote_requests_sample["request_date"].max().strftime("%Y-%m-%d")
    final_report = generate_financial_report(final_date)
    print("\n===== FINAL FINANCIAL REPORT =====")
    print(f"Final Cash: ${final_report['cash_balance']:.2f}")
    print(f"Final Inventory: ${final_report['inventory_value']:.2f}")

    # Save results
    pd.DataFrame(results).to_csv("test_results.csv", index=False)

    # Business Advisor: proactive recommendations based on all handled transactions
    advisor_report = multi_agent_system.run_business_advisor(final_date)
    print("\n===== BUSINESS ADVISOR REPORT =====")
    print(advisor_report)
    with open("business_advisor_report.md", "w", encoding="utf-8") as report_file:
        report_file.write(f"# Business Advisor Report (as of {final_date})\n\n{advisor_report}\n")

    return results


if __name__ == "__main__":
    results = run_test_scenarios()
