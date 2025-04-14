import flwr as fl
from model import build_model
import os
import tensorflow as tf

# Optional: Aggregate accuracy from clients
def weighted_average(metrics):
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    total_examples = sum(num_examples for num_examples, _ in metrics)
    return {"accuracy": sum(accuracies) / total_examples}

class SaveModelStrategy(fl.server.strategy.FedAvg):
    def __init__(self, input_shape, num_classes, **kwargs):
        super().__init__(**kwargs)
        self.input_shape = input_shape
        self.num_classes = num_classes

    def aggregate_fit(self, rnd, results, failures):
        # Aggregate weights using FedAvg
        aggregated_parameters = super().aggregate_fit(rnd, results, failures)

        if aggregated_parameters is not None:
            # Convert Flower parameters to Keras model weights
            final_weights = fl.common.parameters_to_ndarrays(aggregated_parameters)

            # Rebuild and update the Keras model
            model = build_model(self.input_shape, self.num_classes)
            model.set_weights(final_weights)

            # Save the model
            save_path = "flask_app/models/global_model.h5"
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            model.save(save_path)
            print(f"✅ Global model saved to {save_path}")

        return aggregated_parameters

def main():
    # Define your model's input shape and number of output classes
    input_shape = (10,)       # ✅ Update if needed
    num_classes = 2           # ✅ Update if needed

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

    # Start Flower server
    fl.server.start_server(
        server_address="127.0.0.1:8080",
        config=fl.server.ServerConfig(num_rounds=3),
        strategy=strategy,
    )

if __name__ == "__main__":
    main()
