# Candidate Elimination with Detailed Steps

# Dataset
dataset = [
    ["Technical","Senior","Excellent","Good","Urban","Yes"],
    ["Technical","Junior","Excellent","Good","Urban","Yes"],
    ["Non-Technical","Junior","Average","Poor","Rural","No"],
    ["Technical","Senior","Average","Good","Rural","No"],
    ["Technical","Senior","Excellent","Good","Rural","Yes"]
]

attributes = [
    ["Technical", "Non-Technical"],   # ROLE
    ["Senior", "Junior"],              # EXPERIENCE
    ["Excellent", "Average"],          # PERFORMANCE
    ["Good", "Poor"],                  # INTERNET
    ["Urban", "Rural"]                 # LOCATION
]

num_attr = len(attributes)

# Initialize S and G
S = [["Ø"] * num_attr]
G = [["?"] * num_attr]

# Check consistency of a hypothesis with an example
def consistent(hyp, ex):
    for h, e in zip(hyp, ex):
        if h != "?" and h != "Ø" and h != e:
            return False
    return True

# Check if h1 is more general than h2
def more_general(h1, h2):
    more = []
    for x, y in zip(h1, h2):
        more.append(x == "?" or (x != "Ø" and (x == y)))
    return all(more)

print("Initial S:", S)
print("Initial G:", G, "\n")

# Learning
for idx, row in enumerate(dataset):
    x = row[:-1]
    label = row[-1]
    print(f"Example {idx+1}: {x} => {label}")

    if label == "Yes":
        # Remove inconsistent G
        G = [g for g in G if consistent(g, x)]

        # Update S
        newS = []
        for s in S:
            if consistent(s, x):
                newS.append(s)
            else:
                spec = list(s)
                for i in range(num_attr):
                    if spec[i] == "Ø" or (spec[i] != x[i]):
                        spec[i] = x[i]
                newS.append(spec)
        S = newS

        # Keep S consistent with G
        S = [s for s in S if any(more_general(g, s) for g in G)]

    else:  # Negative
        # Remove inconsistent S
        S = [s for s in S if consistent(s, x)]

        # Specialize G
        newG = []
        for g in G:
            if consistent(g, x):
                for i in range(num_attr):
                    if g[i] == "?":
                        for val in attributes[i]:
                            if val != x[i]:
                                spec = list(g)
                                spec[i] = val
                                newG.append(spec)
            else:
                newG.append(g)
        # Remove overly specific hypotheses
        G = []
        for h in newG:
            if not any(more_general(other, h) for other in newG if other != h):
                G.append(h)

    print("S:", S)
    print("G:", G, "\n")

print("Final Specific Boundary S:")
for s in S:
    print(s)

print("\nFinal General Boundary G:")
for g in G:
    print(g)
