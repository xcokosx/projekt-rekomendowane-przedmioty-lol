import random
import json
import argparse
import math
import os


class ItemRanker:

    def __init__(self, n, hidden=64, seed=42):
        self.rand = random.Random(seed)
        self.w1 = [[self.rand.uniform(-0.5, 0.5) for _ in range(n)] for _ in range(hidden)]
        self.w2 = [self.rand.uniform(-0.5, 0.5) for _ in range(hidden)]
        self.b = 0.0
        self.n = n
        self.hidden = hidden

    def relu(self,x):
        return x if x > 0 else 0

    def forward(self, x):

        self.x = x
        self.h = []

        for neuron in self.w1:
            s = sum(w * xi for w, xi in zip(neuron, x))
            self.h.append(self.relu(s))

        self.out_raw = sum(w * h_i for w, h_i in zip(self.w2, self.h)) + self.b
        self.out = 1 / (1 + math.exp(-self.out_raw))

        return self.out
    
    def backward(self, y_true, lr=0.01):

        error = self.out - y_true
        d_out = error * self.out * (1 - self.out)

        d_w2 = [d_out * h for h in self.h]
        d_b = d_out

        d_hidden = [w * d_out for w in self.w2]

        d_hidden = [
            d_h * (1 if h > 0 else 0)
            for d_h, h in zip(d_hidden, self.h)
        ]

        d_w1 = []

        for i in range(len(self.w1)):
            grad_row = []
            for j in range(len(self.x)):
                grad_row.append(d_hidden[i] * self.x[j])
            d_w1.append(grad_row)

        for i in range(len(self.w1)):
            for j in range(len(self.w1[i])):
                self.w1[i][j] -= lr * d_w1[i][j]

        for i in range(len(self.w2)):
            self.w2[i] -= lr * d_w2[i]

        self.b -= lr * d_b

    def save(self, path):
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump({'w1': self.w1, 'w2': self.w2, 'b': self.b, 'n': self.n, 'hidden': self.hidden}, f, ensure_ascii=False)

    @staticmethod
    def load(path):
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        m = ItemRanker(data['n'], hidden=data.get('hidden', len(data.get('w2', []))))
        m.w1 = data.get('w1', [])
        m.w2 = data.get('w2', [])
        m.b = data.get('b', 0.0)
        return m

role_to_id = {
    "TOP": 0,
    "JUNGLE": 1,
    "MID": 2,
    "BOT": 3,
    "SUPPORT": 4
}



with open("./app/data/champions.json", "r", encoding="utf-8") as f:
    champions = json.load(f)

champion_to_id = champion_to_id = {
    champ["id"]: idx
    for idx, champ in enumerate(champions)
}

with open("./app/data/items.json", "r", encoding="utf-8") as f:
    item_data = json.load(f)

all_items = []
item_to_id = {}

for category in item_data.values():

    if not isinstance(category, list):
        continue

    for item in category:

        item_id = item["id"]

        if item_id not in item_to_id:

            item_to_id[item_id] = len(item_to_id)
            all_items.append(item_id)
    #print(f"Znaleziono {len(all_items)} unikalnych przedmiotów")

def encode_champion(name):
    if name not in champion_to_id:
        champion_to_id[name] = len(champion_to_id)
    return champion_to_id[name]

def encode_item(item_id):
    return item_to_id[item_id]

MAX_CHAMPIONS = len(champion_to_id)
MAX_ITEMS = len(item_to_id)

def pad_build(build, max_items=6):

    padded = build[:]

    while len(padded) < max_items:
        padded.append(0)

    return padded[:max_items]

def build_context(sample):

    vec = []

    # champion
    vec.append(encode_champion(sample["champion"])/ MAX_CHAMPIONS)

    # rola
    vec.append(role_to_id[sample["role"]]/ len(role_to_id))

    # ally team 
    allies = sample["ally_team"][:4]
    for a in allies:
        vec.append(encode_champion(a)/ MAX_CHAMPIONS)

    while len(allies) < 4:
        vec.append(-1)
        allies.append(None)

    # enemy team
    enemies = sample["enemy_team"][:5]
    for e in enemies:
        vec.append(encode_champion(e)/ MAX_CHAMPIONS)

    while len(enemies) < 5:
        vec.append(-1)
        enemies.append(None)

    return vec

def extract_matches(data):

    if isinstance(data, list):
        return data

    if isinstance(data, dict):

        if "match_players" in data:
            return data["match_players"]

        if "data" in data:
            return data["data"]

        if "info" in data and "participants" in data["info"]:
            return data["info"]["participants"]

        for v in data.values():
            if isinstance(v, list):
                return v

    raise ValueError(f"Nie można znaleźć listy graczy w danych: {list(data.keys()) if isinstance(data, dict) else type(data)}")
        
def create_samples(match):

    samples = []

    for player in match:

        context = build_context(player)
        real_items = player["items"]
        partial = []

        for real_item in real_items:

            x_pos = context + pad_build(partial) + [encode_item(real_item)/ MAX_ITEMS]
            samples.append((x_pos, 1))

            fake_item = random.choice(all_items)

            while fake_item in real_items or fake_item in partial:
                fake_item = random.choice(all_items)

            x_neg = context + pad_build(partial) + [encode_item(fake_item)/ MAX_ITEMS]
            samples.append((x_neg, 0))

            partial.append(encode_item(real_item)/ MAX_ITEMS)
    
    return samples

def create_samples_with_negatives(match, negatives=1):

    samples = []

    for player in match:

        context = build_context(player)
        real_items = player["items"]
        partial = []

        for real_item in real_items:

            x_pos = context + pad_build(partial) + [encode_item(real_item)/ MAX_ITEMS]
            samples.append((x_pos, 1))

            for _ in range(negatives):

                fake_item = random.choice(all_items)

                while fake_item in real_items or fake_item in partial:

                    fake_item = random.choice(all_items)

                x_neg = context + pad_build(partial) + [encode_item(fake_item)/ MAX_ITEMS]
                samples.append((x_neg, 0))

            partial.append(encode_item(real_item)/ MAX_ITEMS)

    return samples

def recommend(model, context):

    build = []

    for step in range(6):

        best_item = None
        best_score = -1

        for item in all_items:
            
            if item in build:
                continue

            x = context + pad_build(build) + [encode_item(item)/ MAX_ITEMS]
            score = model.forward(x)

            if score > best_score:
                best_score = score
                best_item = item

        build.append(best_item)

    return build


def train(model, samples, epochs=20, lr=0.01, batch_size=32):
    metrics = []
    for epoch in range(epochs):
        random.shuffle(samples)
        total_loss = 0.0
        for i in range(0, len(samples), batch_size):
            batch = samples[i:i+batch_size]
            for x, y in batch:
                pred = model.forward(x)
                model.backward(y, lr)
                total_loss += (pred - y) ** 2

        avg_loss = total_loss / len(samples) if samples else 0
        print(f"epoch {epoch} loss {avg_loss:.6f}")
        metrics.append({"epoch": epoch, "loss": avg_loss})

    return metrics


def save_mappings(path, champion_map, item_map):
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    # convert item_map keys to strings for JSON
    item_map_serial = {str(k): v for k, v in item_map.items()}
    with open(path, 'w', encoding='utf-8') as f:
        json.dump({'champion_to_id': champion_map, 'item_to_id': item_map_serial}, f, ensure_ascii=False)


def load_mappings(path):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    champ = data.get('champion_to_id', {})
    item_raw = data.get('item_to_id', {})
    # convert item keys back to ints when possible
    item = {}
    for k, v in item_raw.items():
        try:
            ik = int(k)
        except:
            ik = k
        item[ik] = v
    return champ, item


#with open("./app/data/preprocessed_match_champion_data.json", "r", encoding="utf-8") as f:
def main():
    parser = argparse.ArgumentParser(description='Train item ranker')
    parser.add_argument('--data', default='./app/data/preprocessed_match_champion_data.json')
    #parser.add_argument('--data', default='./app/data/preprocessed_match_champion_data_testing.json')  # DEBUGGING PURPOSES, SMALLER DATASET
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--batch', type=int, default=32)
    parser.add_argument('--model-out', default='./app/model.json')
    parser.add_argument('--mappings-out', default='./app/mappings.json')
    parser.add_argument('--metrics-out', default='./app/metrics.json')
    parser.add_argument('--negatives', type=int, default=1)
    args = parser.parse_args()

    with open(args.data, 'r', encoding='utf-8') as f:
        data = json.load(f)

    match = extract_matches(data)
    samples = create_samples_with_negatives(match, negatives=args.negatives)

    if not samples:
        print('Brak próbek do treningu')
        return

    model = ItemRanker(n=len(samples[0][0]))
    metrics = train(model, samples, epochs=args.epochs, lr=args.lr, batch_size=args.batch)

    model.save(args.model_out)
    save_mappings(args.mappings_out, champion_to_id, item_to_id)

    if args.metrics_out:
        os.makedirs(os.path.dirname(args.metrics_out) or '.', exist_ok=True)
        with open(args.metrics_out, 'w', encoding='utf-8') as mf:
            json.dump({'metrics': metrics}, mf, ensure_ascii=False)

    print(f'Model saved to {args.model_out}, mappings to {args.mappings_out}, metrics to {args.metrics_out}')


if __name__ == '__main__':
    main()