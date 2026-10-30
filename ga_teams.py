"""
จัดทีมกีฬาสี 2 สีด้วย Genetic Algorithm (บทที่ 07 Genetic Algorithms)

Chromosome : array 0/1 ยาวเท่าจำนวนพนักงาน, gene ที่ i = สีของพนักงานคนที่ i (0 = แดง, 1 = น้ำเงิน)
Fitness    : ผลรวม penalty ถ่วงน้ำหนัก -> ยิ่งน้อยยิ่งดี (แนวเดียวกับ SSE ใน curve fitting)
             ข้อบังคับ (size, dept_cap) น้ำหนักสูง, ความยุติธรรม (gender, age, body) น้ำหนักเท่ากัน
Selection  : topk (hard, เก็บครึ่งบนแบบตัวอย่างอาจารย์) หรือ roulette (soft) + elitism
Reproduction: crossover (single / two / uniform) + mutation (สลับสีพนักงาน 1 คู่)

โค้ดวาดกราฟอยู่ใน ga_sports_teams.ipynb
"""
import csv

import numpy as np

TEAM_NAMES = ["Red", "Blue"]
AGE_BINS = [30, 40, 50]  # แบ่งช่วงอายุ <30, 30-39, 40-49, 50+
AGE_LABELS = ["<30", "30-39", "40-49", "50+"]

# น้ำหนักของแต่ละเงื่อนไขใน fitness
# ข้อบังคับให้น้ำหนักสูง (penalty method) -> GA ทำให้ผ่านข้อบังคับก่อน แล้วค่อยปรับความยุติธรรม
WEIGHTS = {
    "size": 10.0,      # [ข้อบังคับ] จำนวนคนสองสีเท่ากัน
    "dept_cap": 10.0,  # [ข้อบังคับ] แต่ละแผนกมีในสีหนึ่งได้ไม่เกิน ceil(n_d / 2) คน
    "gender": 1.0,     # [ยุติธรรม] ชาย/หญิงแต่ละสีเท่ากัน
    "age": 1.0,        # [ยุติธรรม] การกระจายช่วงอายุ + อายุเฉลี่ย
    "body": 1.0,       # [ยุติธรรม] น้ำหนักและส่วนสูงเฉลี่ยใกล้กัน
}


# ---------------------------------------------------------------- data
def load_hr(path="HR.csv"):
    """อ่าน HR.csv -> dict ของ array: gender, department, age, weight, height"""
    with open(path, newline="", encoding="utf-8-sig") as f:
        rows = [[c.strip() for c in r] for r in csv.reader(f) if r]
    header, rows = rows[0], rows[1:]
    col = {name: i for i, name in enumerate(header)}
    data = {k: np.array([r[col[k]] for r in rows]) for k in ("gender", "department")}
    for k in ("age", "weight", "height"):
        data[k] = np.array([float(r[col[k]]) for r in rows])
    return data


def one_hot(values):
    cats = np.unique(values)
    return (values[:, None] == cats[None, :]).astype(float), cats


class Problem:
    """เก็บข้อมูลที่ใช้คำนวณ fitness แบบ vectorize ทั้ง population ในครั้งเดียว"""

    def __init__(self, data):
        self.data = data
        self.n = len(data["gender"])
        self.g_onehot, self.genders = one_hot(data["gender"])
        self.d_onehot, self.depts = one_hot(data["department"])
        self.age_group = np.digitize(data["age"], AGE_BINS)
        self.a_onehot, _ = one_hot(self.age_group)

        self.dept_size = self.d_onehot.sum(0)
        self.dept_cap = np.ceil(self.dept_size / 2)  # เพดานตามสัดส่วนจริง (ครึ่งหนึ่งของแผนก ปัดขึ้น)
        # จำนวนคนเกินเพดานสูงสุดที่เป็นไปได้ (ทุกคนอยู่สีเดียว) ใช้ปรับสเกลให้อยู่ในช่วง 0-1
        self.max_excess = (self.dept_size - self.dept_cap).sum()

        # z-score ของ อายุ/น้ำหนัก/ส่วนสูง ให้ทุกตัวแปรมีสเกลเท่ากัน
        num = np.column_stack([data["age"], data["weight"], data["height"]])
        self.z = (num - num.mean(0)) / num.std(0)

    def team_counts(self, pop, onehot):
        """จำนวนคนต่อกลุ่มในสีน้ำเงินและแดง: shape (P, จำนวนกลุ่ม)"""
        blue = pop @ onehot
        return onehot.sum(0) - blue, blue

    def penalties(self, pop):
        """pop: (P, n) ของ 0/1 -> dict ของ penalty แต่ละเงื่อนไข shape (P,) ทุกค่าอยู่ในช่วง 0-1"""
        pop = np.atleast_2d(pop)

        def balance(onehot):
            # ผลต่างจำนวนคนระหว่าง 2 สีต่อกลุ่ม หารด้วยขนาดกลุ่ม: 0 = แบ่งครึ่งพอดี, 1 = อยู่สีเดียวทั้งกลุ่ม
            # กลุ่มที่มีจำนวนคี่ต่างกันอย่างน้อย 1 คนเสมอ จึงหักส่วนนั้นออกให้ค่าดีที่สุดเป็น 0
            red, blue = self.team_counts(pop, onehot)
            size = onehot.sum(0)
            return ((np.abs(blue - red) - size % 2) / size) ** 2

        n_blue = pop.sum(1)
        size = ((np.abs(2 * n_blue - self.n) - self.n % 2) / self.n) ** 2

        red, blue = self.team_counts(pop, self.d_onehot)
        excess = np.maximum(red - self.dept_cap, 0) + np.maximum(blue - self.dept_cap, 0)
        dept_cap = excess.sum(1) / self.max_excess

        gender = balance(self.g_onehot).mean(1)

        n_b = n_blue.clip(1, self.n - 1)[:, None]
        mean_diff = (pop @ self.z) / n_b - ((1 - pop) @ self.z) / (self.n - n_b)  # (P, 3)
        age = 0.5 * balance(self.a_onehot).mean(1) + 0.5 * mean_diff[:, 0] ** 2
        body = (mean_diff[:, 1:] ** 2).mean(1)

        return {"size": size, "dept_cap": dept_cap, "gender": gender, "age": age, "body": body}

    def fitness(self, pop):
        p = self.penalties(pop)
        return sum(WEIGHTS[k] * v for k, v in p.items())


# ---------------------------------------------------------------- GA operators
def select(pop, fit, n_parents, method, rng, k=None):
    if method == "topk":
        # Hard selection: สุ่มพ่อแม่จาก k ตัวที่ดีที่สุดเท่านั้น
        best = np.argsort(fit)[:k]
        idx = rng.choice(best, n_parents)
    else:
        # Soft selection (roulette wheel): โอกาสถูกเลือก ∝ 1/fitness (fitness ยิ่งน้อยยิ่งดี)
        score = 1.0 / (fit + 1e-12)
        idx = rng.choice(len(pop), n_parents, p=score / score.sum())
    return pop[idx]


def crossover(p1, p2, method, rng):
    n = p1.shape[1]
    pos = np.arange(n)
    if method == "single":
        cut = rng.integers(1, n, size=(len(p1), 1))
        mask = pos < cut
    elif method == "two":
        a = np.sort(rng.integers(1, n, size=(len(p1), 2)), axis=1)
        mask = (pos >= a[:, :1]) & (pos < a[:, 1:])
    else:  # uniform: สุ่มทีละ gene ว่ามาจากพ่อหรือแม่
        mask = rng.random(p1.shape) < 0.5
    return np.where(mask, p1, p2)


def mutate(pop, rate, rng):
    # สลับสีพนักงาน 1 คู่ (แดง 1 คน <-> น้ำเงิน 1 คน) ขนาดทีมจึงไม่เปลี่ยน
    pop = pop.copy()
    for i in np.where(rng.random(len(pop)) < rate)[0]:
        reds, blues = np.flatnonzero(pop[i] == 0), np.flatnonzero(pop[i] == 1)
        if len(reds) and len(blues):
            pop[i, rng.choice(reds)] = 1
            pop[i, rng.choice(blues)] = 0
    return pop


def run_ga(problem, pop_size=100, generations=300, selection="topk", crossover_method="single",
           mutation_rate=0.8, elite=2, seed=0, verbose=False, print_every=10):
    """
    คืนค่า (best chromosome, history)
    history: best / mean fitness, penalty ของตัวที่ดีที่สุด และ chromosome ที่ดีที่สุดของทุกรุ่น
    - topk     : เก็บครึ่งบนไว้ทั้งหมด แล้วแทนครึ่งล่างด้วยลูกของครึ่งบน (แบบตัวอย่างอาจารย์)
    - roulette : เก็บ elite ตัวที่ดีที่สุดไว้ ที่เหลือเป็นลูกจาก roulette wheel
    """
    rng = np.random.default_rng(seed)
    pop = rng.integers(0, 2, size=(pop_size, problem.n)).astype(np.int8)
    keep = pop_size // 2 if selection == "topk" else elite
    history = {"best": [], "mean": [], "parts": {k: [] for k in WEIGHTS}, "best_chrom": []}

    for gen in range(generations + 1):
        fit = problem.fitness(pop)
        order = np.argsort(fit)
        pop, fit = pop[order], fit[order]

        history["best"].append(fit[0])
        history["mean"].append(fit.mean())
        for k, v in problem.penalties(pop[:1]).items():
            history["parts"][k].append(v[0])
        history["best_chrom"].append(pop[0].copy())
        if verbose and gen % print_every == 0:
            print(f"gen {gen:4d}  fitness = {fit[0]:.6f}")
        if gen == generations:
            break

        n_child = pop_size - keep
        p1 = select(pop, fit, n_child, selection, rng, k=pop_size // 2)
        p2 = select(pop, fit, n_child, selection, rng, k=pop_size // 2)
        children = mutate(crossover(p1, p2, crossover_method, rng), mutation_rate, rng)
        pop = np.vstack([pop[:keep], children])

    history = {k: (np.array(v) if k != "parts" else {p: np.array(x) for p, x in v.items()})
               for k, v in history.items()}
    return pop[0], history


# ---------------------------------------------------------------- report
def team_table(problem, chrom):
    """สรุปแต่ละสี: list ของ dict (ใช้แสดงเป็นตาราง)"""
    d = problem.data
    rows = []
    for t, name in enumerate(TEAM_NAMES):
        m = chrom == t
        row = {"สี": name, "จำนวน": int(m.sum())}
        for g in problem.genders:
            row[g] = int((d["gender"][m] == g).sum())
        row["อายุเฉลี่ย"] = round(d["age"][m].mean(), 2)
        row["น้ำหนักเฉลี่ย"] = round(d["weight"][m].mean(), 2)
        row["ส่วนสูงเฉลี่ย"] = round(d["height"][m].mean(), 2)
        row["จำนวนแผนก"] = len(np.unique(d["department"][m]))
        rows.append(row)
    return rows


def check_constraints(problem, chrom):
    """ตรวจข้อบังคับ: ขนาดทีม และไม่มีแผนกไหนเกินเพดานในสีใด"""
    red, blue = problem.team_counts(chrom[None, :], problem.d_onehot)
    over = [(dep, int(r), int(b), int(c)) for dep, r, b, c in
            zip(problem.depts, red[0], blue[0], problem.dept_cap) if r > c or b > c]
    return {"red": int((chrom == 0).sum()), "blue": int((chrom == 1).sum()), "over_cap": over}


def export_teams(problem, chrom, path):
    d = problem.data
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["id", "gender", "department", "age", "weight", "height", "team"])
        for i in range(problem.n):
            w.writerow([i + 1, d["gender"][i], d["department"][i], int(d["age"][i]),
                        int(d["weight"][i]), int(d["height"][i]), TEAM_NAMES[chrom[i]]])
