import matplotlib.pyplot as plt

# Read accuracy data from global_accuracy.txt
rounds, accuracies = [], []
try:
    with open("global_accuracy.txt", "r") as f:
        for line in f:
            rnd, acc = line.strip().split(",")
            rounds.append(int(rnd))
            accuracies.append(float(acc))
except FileNotFoundError:
    print("Error: global_accuracy.txt not found")
    exit(1)
except ValueError:
    print("Error: Invalid format in global_accuracy.txt")
    exit(1)

# Plot the data
plt.figure(figsize=(8, 6))
plt.plot(rounds, accuracies, marker="o", linestyle="-", color="b")
plt.xlabel("Round")
plt.ylabel("Global Accuracy")
plt.title("Federated Learning: Stress Detection Accuracy")
plt.grid(True)
plt.ylim(0, 1.1)  # Accuracy range 0–1, with padding
plt.xticks(rounds)  # Show all rounds
plt.savefig("accuracy_plot.png", dpi=300)
plt.show()