import json
import random
import os
from faker import Faker

NUM_TRAIN = 900
NUM_DEV = 100
OUTPUT_DIR = "data"
FAKE = Faker('en_US')
random.seed(42)

def add_noise_to_digits(val):
    """
    Injects realistic STT formatting noise into digit sequences.
    Ex: "5551234" -> "5 55- 1 2 34"
    """
    chars = list(val)
    noisy_chars = []
    for c in chars:
        if c.isdigit():
            # 30% chance to add a separator before the digit
            if random.random() < 0.3:
                noisy_chars.append(random.choice([' ', '-', ' . ']))
            noisy_chars.append(c)
        else:
            noisy_chars.append(c)
    return "".join(noisy_chars)

def generate_example(id_num):
    parts = []
    entities = []
    current_idx = 0
    
    def add_text(text):
        nonlocal current_idx
        parts.append(text)
        current_idx += len(text)
        
    def add_entity(text, label, pii_status):
        nonlocal current_idx
        start = current_idx
        parts.append(text)
        end = current_idx + len(text)
        current_idx = end
        
        entities.append({
            "start": start, "end": end, "label": label, "pii": pii_status
        })

    # Added 'loc' to this list so Non-PII data is generated
    template_type = random.choice(['cc', 'phone', 'email', 'mixed', 'name', 'date', 'garbage', 'loc'])
    
    if template_type == 'cc':
        intro = random.choice(["my card is ", "number is ", "write down "])
        add_text(intro)
        cc_val = add_noise_to_digits(FAKE.credit_card_number())
        add_entity(cc_val, "CREDIT_CARD", True)
        
    elif template_type == 'phone':
        intro = random.choice(["call ", "dial ", "reach me at "])
        add_text(intro)
        phone_val = add_noise_to_digits(FAKE.phone_number())
        add_entity(phone_val, "PHONE", True)
        
    elif template_type == 'email':
        add_text("email is ")
        email_val = FAKE.email()
        if random.random() < 0.5:
            email_val = email_val.replace("@", " at ").replace(".", " dot ")
        add_entity(email_val, "EMAIL", True)
        
    elif template_type == 'mixed':
        add_text("i am ")
        add_entity(FAKE.name(), "PERSON_NAME", True)
        add_text(" and phone is ")
        phone_val = add_noise_to_digits(FAKE.phone_number())
        add_entity(phone_val, "PHONE", True)

    elif template_type == 'name':
        add_text("hello this is ")
        name = FAKE.name()
        if random.random() < 0.3:
            name = name.replace(" ", " " + random.choice("ABCDEFGHIJKLMNOPQRSTUVWXYZ") + ". ")
        add_entity(name, "PERSON_NAME", True)

    elif template_type == 'date':
        add_text("date is ")
        add_entity(FAKE.date(), "DATE", True)

    elif template_type == 'loc':
        # --- THIS FIXES THE NON-PII SCORE ---
        add_text("i live in ")
        add_entity(FAKE.city(), "CITY", False)
        add_text(" specifically at ")
        add_entity(FAKE.address(), "LOCATION", False)

    elif template_type == 'garbage':
        add_text(FAKE.text(max_nb_chars=50))

    full_text = "".join(parts).lower()
    
    return {"id": f"utt_{id_num:04d}", "text": full_text, "entities": entities}

def main():
    print("Generating FINAL ROBUST data (PII + Non-PII)...")
    all_data = [generate_example(i) for i in range(NUM_TRAIN + NUM_DEV)]
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(f"{OUTPUT_DIR}/train_gen.jsonl", 'w') as f:
        for item in all_data[:NUM_TRAIN]: f.write(json.dumps(item) + '\n')
    with open(f"{OUTPUT_DIR}/dev_gen.jsonl", 'w') as f:
        for item in all_data[NUM_TRAIN:]: f.write(json.dumps(item) + '\n')

if __name__ == "__main__":
    main()