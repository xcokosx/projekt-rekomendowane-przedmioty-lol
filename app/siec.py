import random
import json
import argparse
import math
import os
import hashlib
global MAX_CHAMPIONS, MAX_ITEMS, MODE
MODE = 'train'

class ItemRanker:

    def __init__(self, n, hidden=64, seed=42):
        self.rand = random.Random(seed)
        self.w1 = [[self.rand.uniform(-0.1, 0.1) for _ in range(n)] for _ in range(hidden)]
        self.w2 = [self.rand.uniform(-0.1, 0.1) for _ in range(hidden)]
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

        if self.out_raw >= 700:
            self.out = 1.0
        elif self.out_raw <= -700:
            self.out = 0.0
        else:
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


def normalize_name(s: str):
    if not s:
        return ''
    return ''.join(ch.lower() for ch in s if ch.isalnum())


# map normalized champion names -> index
champion_to_id = {
    normalize_name(champ.get('name') or str(champ.get('id'))): idx
    for idx, champ in enumerate(champions)
}

# helper: map champion id (as str) -> name from data
id_to_champion_name = {str(champ.get('id')): champ.get('name') for champ in champions}

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

# build id -> name map for nicer output
item_id_to_name = {}
for category in item_data.values():
    if not isinstance(category, list):
        continue
    for item in category:
        if isinstance(item, dict):
            iid = item.get('id')
            name = item.get('name') or item.get('displayName') or str(iid)
            item_id_to_name[iid] = name
        else:
            # fallback when item is just an id
            try:
                item_id_to_name[int(item)] = str(item)
            except:
                pass

# recommended candidates: only completed items (avoid starter/component/misc/full_list)
completed_items = []
if isinstance(item_data, dict) and 'completed_item' in item_data:
    for it in item_data.get('completed_item', []):
        try:
            completed_items.append(it['id'])
        except Exception:
            try:
                completed_items.append(int(it))
            except Exception:
                pass


def encode_champion(name):
    global MAX_CHAMPIONS
    key = normalize_name(name)
    if key in champion_to_id:
        return champion_to_id[key]

    # in training mode allow dynamic assignment, but in recommend avoid mutating maps
    if MODE == 'train':
        new_idx = len(champion_to_id)
        champion_to_id[key] = new_idx
        MAX_CHAMPIONS = len(champion_to_id) or 1
        return new_idx

    # recommend/inference: unknown champion -> fallback index 0
    return 0

def encode_item(item_id):
    global MAX_ITEMS
    # Reserve index 0 for padding/unknown. Real items -> index+1
    if item_id in item_to_id:
        return item_to_id[item_id] + 1

    if MODE == 'train':
        new_idx = len(item_to_id)
        item_to_id[item_id] = new_idx
        MAX_ITEMS = (len(item_to_id) + 1) or 1
        return new_idx + 1

    # recommend/inference: unknown -> padding/unknown index 0
    return 0

MAX_CHAMPIONS = len(champion_to_id)
MAX_ITEMS = len(item_to_id) + 1

def pad_build(build, max_items=6):

    padded = build[:]

    # padding uses 0 (reserved for empty slot / unknown)
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
        vec.append(0)
        allies.append(None)

    # enemy team
    enemies = sample["enemy_team"][:5]
    for e in enemies:
        vec.append(encode_champion(e)/ MAX_CHAMPIONS)

    while len(enemies) < 5:
        vec.append(0)
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
        partial_ids = []

        for real_item in real_items:

            encoded_partial = [encode_item(pid)/ MAX_ITEMS for pid in partial_ids]
            x_pos = context + pad_build(encoded_partial) + [encode_item(real_item)/ MAX_ITEMS]
            samples.append((x_pos, 1))

            fake_item = random.choice(all_items)

            while fake_item in real_items or fake_item in partial_ids:
                fake_item = random.choice(all_items)

            x_neg = context + pad_build(encoded_partial) + [encode_item(fake_item)/ MAX_ITEMS]
            samples.append((x_neg, 0))

            partial_ids.append(real_item)
    
    return samples

def create_samples_with_negatives(match, negatives=1):

    samples = []

    for player in match:

        context = build_context(player)
        real_items = player["items"]
        partial_ids = []

        for real_item in real_items:

            encoded_partial = [encode_item(pid)/ MAX_ITEMS for pid in partial_ids]
            x_pos = context + pad_build(encoded_partial) + [encode_item(real_item)/ MAX_ITEMS]
            samples.append((x_pos, 1))

            for _ in range(negatives):

                fake_item = random.choice(all_items)

                while fake_item in real_items or fake_item in partial_ids:

                    fake_item = random.choice(all_items)

                x_neg = context + pad_build(encoded_partial) + [encode_item(fake_item)/ MAX_ITEMS]
                samples.append((x_neg, 0))

            partial_ids.append(real_item)

    return samples

def recommend(model, context, seed=None, deterministic=False, seen_positive_items=None, debug=False, top_k=10):

    if seed is not None:
        random.seed(seed)

    build = []

    # Prefer items that are both completed and seen during training (if available)
    if seen_positive_items:
        candidates = [c for c in completed_items if c in item_to_id and c in seen_positive_items]
    else:
        candidates = [c for c in completed_items if c in item_to_id] if completed_items else all_items

    # Fallbacks if filtering removed all candidates
    if not candidates:
        candidates = [c for c in completed_items if c in item_to_id] or all_items

    # avoid deterministic id-order bias when not explicitly requested
    if not deterministic:
        random.shuffle(candidates)

    for step in range(6):

        best_item = None
        best_score = -1

        scores_list = []
        for item in candidates:
            if item in build:
                continue

            encoded_build = [encode_item(b)/ MAX_ITEMS for b in build]
            x = context + pad_build(encoded_build) + [encode_item(item)/ MAX_ITEMS]
            score = model.forward(x)

            # tiny noise to break exact-score ties
            if not deterministic:
                score += random.random() * 1e-9
            else:
                # deterministic small tie-breaker based on item id
                try:
                    b = str(item).encode('utf-8')
                    h = hashlib.md5(b).hexdigest()[:8]
                    noise = int(h, 16) % 100000 / 1e12
                    score += noise
                except Exception:
                    pass

            scores_list.append((item, score))

            if score > best_score:
                best_score = score
                best_item = item

        # if no candidate found (pool too small), try fallback to all_items
        if best_item is None:
            # allow fallback to any item from data (even if not in loaded item_to_id)
            fallback = [c for c in all_items if c not in build]
            for item in fallback:
                encoded_build = [encode_item(b)/ MAX_ITEMS for b in build]
                x = context + pad_build(encoded_build) + [encode_item(item)/ MAX_ITEMS]
                score = model.forward(x)
                if not deterministic:
                    score += random.random() * 1e-9
                if score > best_score:
                    best_score = score
                    best_item = item

        # if still none, try deterministic fallback pick from all_items (avoid skipping steps)
        if best_item is None:
            fallback_candidates = [c for c in all_items if c not in build]
            if fallback_candidates:
                if deterministic:
                    # deterministic choice: smallest stable hash
                    def stable_key(x):
                        h = hashlib.md5(str(x).encode('utf-8')).hexdigest()
                        return int(h[:8], 16)
                    best_item = sorted(fallback_candidates, key=stable_key)[0]
                else:
                    best_item = random.choice(fallback_candidates)
            else:
                break

        # debug: print top-k candidates for this step
        if debug:
            # ensure scores_list contains fallback scores as well
            if not scores_list:
                # recompute scores_list from fallback
                scores_list = []
                for item in (fallback if 'fallback' in locals() else candidates):
                    if item in build:
                        continue
                    encoded_build = [encode_item(b)/ MAX_ITEMS for b in build]
                    sc = model.forward(context + pad_build(encoded_build) + [encode_item(item)/ MAX_ITEMS])
                    scores_list.append((item, sc))

            scores_sorted = sorted(scores_list, key=lambda x: x[1], reverse=True)[:top_k]
            print(f"[recommend debug] step={step} candidates={len(candidates)} top={[(item_id_to_name.get(i,str(i)), round(s,6)) for i,s in scores_sorted]}")

        build.append(best_item)
    #print(f"Recommended build (ids): {build}")
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


def save_mappings(path, champion_map, item_map, seen_positive_items=None):
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    # convert item_map keys to strings for JSON
    item_map_serial = {str(k): v for k, v in item_map.items()}
    with open(path, 'w', encoding='utf-8') as f:
        json.dump({'champion_to_id': champion_map, 'item_to_id': item_map_serial, 'seen_positive_items': list(seen_positive_items) if seen_positive_items is not None else []}, f, ensure_ascii=False)


def load_mappings(path):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    champ = data.get('champion_to_id', {})
    item_raw = data.get('item_to_id', {})
    seen = data.get('seen_positive_items', [])
    # convert item keys back to ints when possible
    item = {}
    for k, v in item_raw.items():
        try:
            ik = int(k)
        except:
            ik = k
        item[ik] = v
    return champ, item, seen


#with open("./app/data/preprocessed_match_champion_data.json", "r", encoding="utf-8") as f:
def main():
    parser = argparse.ArgumentParser(description='Train item ranker')
    parser.add_argument('--mode', choices=['train', 'recommend'], default='train')
    parser.add_argument('--data', default='./app/data/preprocessed_match_champion_data.json')
    #parser.add_argument('--data', default='./app/data/preprocessed_match_champion_data_testing.json')  # DEBUGGING PURPOSES, SMALLER DATASET
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--batch', type=int, default=32)
    parser.add_argument('--model-out', default='./app/model.json')
    parser.add_argument('--mappings-out', default='./app/mappings.json')
    parser.add_argument('--metrics-out', default='./app/metrics.json')
    parser.add_argument('--negatives', type=int, default=1)
    parser.add_argument('--continue', '--continue-training', dest='continue_training', action='store_true', help='Continue training from existing model if present')
    # recommend mode args
    parser.add_argument('--model-in', default='./app/model.json')
    parser.add_argument('--mappings-in', default='./app/mappings.json')
    parser.add_argument('--champion', default='Ahri')
    parser.add_argument('--role', default='MID')
    parser.add_argument('--ally', default='')
    parser.add_argument('--enemy', default='')
    parser.add_argument('--seed', type=int, default=None, help='Optional RNG seed for deterministic recommend')
    parser.add_argument('--deterministic', action='store_true', help='Disable tie-breaking randomness in recommend')
    parser.add_argument('--debug', action='store_true', help='Print debug info for recommend')
    args = parser.parse_args()
    global MODE
    MODE = args.mode

    if args.mode == 'train':
        with open(args.data, 'r', encoding='utf-8') as f:
            data = json.load(f)

        match = extract_matches(data)
        samples = create_samples_with_negatives(match, negatives=args.negatives)

        if not samples:
            print('Brak próbek do treningu')
            return

        # initialize or continue training from existing model
        if args.continue_training and os.path.exists(args.model_out):
            try:
                model = ItemRanker.load(args.model_out)
                champ_map, item_map = load_mappings(args.mappings_out)
                if champ_map:
                    champion_to_id.clear(); champion_to_id.update(champ_map)
                if item_map:
                    item_to_id.clear(); item_to_id.update(item_map)

                # recompute maxima after loading mappings
                MAX_CHAMPIONS = len(champion_to_id) or 1
                MAX_ITEMS = len(item_to_id) + 1 or 1

                # verify input size matches; if not, fall back to new model
                if getattr(model, 'n', None) != len(samples[0][0]):
                    print('Loaded model input size mismatch — starting from scratch')
                    model = ItemRanker(n=len(samples[0][0]))
            except Exception as e:
                print('Błąd ładowania modelu — zaczynam od zera:', e)
                model = ItemRanker(n=len(samples[0][0]))
        else:
            model = ItemRanker(n=len(samples[0][0]))

        metrics = train(model, samples, epochs=args.epochs, lr=args.lr, batch_size=args.batch)

        model.save(args.model_out)
        # collect items seen as positives in the provided match data
        seen_positive_items = set()
        try:
            for player in match:
                for it in player.get('items', []):
                    seen_positive_items.add(it)
        except Exception:
            seen_positive_items = set()

        save_mappings(args.mappings_out, champion_to_id, item_to_id, seen_positive_items=seen_positive_items)

        if args.metrics_out:
            os.makedirs(os.path.dirname(args.metrics_out) or '.', exist_ok=True)
            with open(args.metrics_out, 'w', encoding='utf-8') as mf:
                json.dump({'metrics': metrics}, mf, ensure_ascii=False)

        print(f'Model saved to {args.model_out}, mappings to {args.mappings_out}, metrics to {args.metrics_out}')

    else:  # recommend
        try:
            model = ItemRanker.load(args.model_in)
        except Exception as e:
            print('Nie można załadować modelu:', e)
            return

        # load mappings produced at training time to ensure consistent encoding
        try:
            champ_map, item_map, seen = load_mappings(args.mappings_in)
        except Exception as e:
            print('Nie można załadować mapowań:', e)
            champ_map, item_map, seen = {}, {}, []

        # update global maps to match training
        if champ_map:
            # convert stored champion keys (which may be numeric ids or names)
            normalized_champ_map = {}
            for k, v in champ_map.items():
                try:
                    # if key corresponds to champion id as string
                    name = id_to_champion_name.get(str(k))
                    if name:
                        nk = normalize_name(name)
                    else:
                        nk = normalize_name(str(k))
                except Exception:
                    nk = normalize_name(str(k))
                normalized_champ_map[nk] = v

            champion_to_id.clear()
            champion_to_id.update(normalized_champ_map)

        if item_map:
            item_to_id.clear()
            item_to_id.update(item_map)

        # recompute maxima
        MAX_CHAMPIONS = len(champion_to_id) or 1
        MAX_ITEMS = len(item_to_id) + 1 or 1

        def tidy_name(s: str):
            if not s:
                return ''
            s = s.strip()
            if ' ' in s:
                return ''.join(word.capitalize() for word in s.split())
            return s.capitalize()

        champ = tidy_name(args.champion) if args.champion else None
        allies = [tidy_name(x) for x in args.ally.split(',') if x.strip()]
        enemies = [tidy_name(x) for x in args.enemy.split(',') if x.strip()]

        sample = {
            'champion': champ,
            'role': (args.role or 'MID').upper(),
            'ally_team': allies,
            'enemy_team': enemies,
            'items': []
        }

        context = build_context(sample)
        build = recommend(model, context, seed=args.seed, deterministic=args.deterministic, seen_positive_items=seen, debug=args.debug)

        build_named = [item_id_to_name.get(item, str(item)) for item in build]

        print('Recommended build (ids):', build)
        print('Recommended build (names):', build_named)


if __name__ == '__main__':
    main()
