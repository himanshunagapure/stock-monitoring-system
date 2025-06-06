import re

def normalize_condition_text(text):
    text = text.lower().replace(" ", "")
    
    # Patterns for 'above' conditions
    above_keywords = ["above", "greaterthan", "greater", "morethan", "more", "exceed", "higher", ">", "gt"]
    for keyword in above_keywords:
        if keyword in text:
            return "above"

    # Patterns for 'below' conditions
    below_keywords = ["below", "lessthan", "less", "smaller", "under", "<", "lt"]
    for keyword in below_keywords:
        if keyword in text:
            return "below"

    return "unknown"

def check_condition(price, rule):
    print("🤖 Checking condition rule:")
    print(rule)

    raw_condition = rule.get("condition", "")
    target_price = rule.get("target_price")

    if target_price is None:
        print("❌ No target price provided.")
        return False

    # Normalize condition text to "above" or "below"
    condition = normalize_condition_text(raw_condition)

    if condition == "above" and price > target_price:
        print(f"✔️ Condition Met: '{raw_condition}'")
        return True
    if condition == "below" and price < target_price:
        print(f"✔️ Condition Met: '{raw_condition}'")
        return True

    print(f"❌ Unrecognized or unmatched condition: '{raw_condition}'")
    return False


''' 
def check_condition(price, rule):
    print("🤫🤫🤫Check condition rule 🤫🤫")
    print(rule)
    condition = rule["condition"].lower()
    value = rule["target_price"]

    if condition == "above" and price > value:
        return True
    if condition == "below" and price < value:
        return True
    return False
'''