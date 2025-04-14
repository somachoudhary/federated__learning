import flwr as fl
from model import build_model
import os
import tensorflow as tf

# Suppress TensorFlow warnings (optional)
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

def weighted_average(metrics):
    """Aggregate client accuracies weighted by number of examples."""
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    total_examples = sum(num_examples for num_examples, _ in metrics)
    return {"accuracy": sum(accuracies) / total_examples}

class SaveModelStrategy(fl.server.strategy.FedAvg):
    def __init__(self, input_shape, num_classes, **kwargs):
        super().__init__(**kwargs)
        self.input_shape = input_shape
        self.num_classes = num_classes

    def aggregate_fit(self, rnd, results, failures):
        aggregated_result = super().aggregate_fit(rnd, results, failures)
        if aggregated_result is not None:
            aggregated_parameters, _ = aggregated_result
            final_weights = fl.common.parameters_to_ndarrays(aggregated_parameters)
            model = build_model(input_shape=self.input_shape, binary=(self.num_classes == 2), num_classes=self.num_classes)
            model.set_weights(final_weights)
            save_path = "models/global_model.weights.h5"  # Updated extension
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            model.save_weights(save_path)
            print(f"✅ Round {rnd}: Global model saved to {save_path}")
        return aggregated_result

    def aggregate_evaluate(self, rnd, results, failures):
        aggregated_result = super().aggregate_evaluate(rnd, results, failures)
        if aggregated_result is not None:
            loss, metrics = aggregated_result
            print(f"🌟 Round {rnd}: Global accuracy = {metrics['accuracy']:.4f}")
            with open("global_accuracy.txt", "a") as f:
                f.write(f"{rnd},{metrics['accuracy']}\n")
        return aggregated_result

def main():
    input_shape = (5,)
    num_classes = 2

    strategy = SaveModelStrategy(
        input_shape=input_shape,
        num_classes=num_classes,
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=2,
        min_evaluate_clients=2,
        min_available_clients=2,
        evaluate_metrics_aggregation_fn=weighted_average,
    )

    fl.server.start_server(
        server_address="127.0.0.1:8080",
        config=fl.server.ServerConfig(num_rounds=3),
        strategy=strategy,
    )

if __name__ == "__main__":
    main()