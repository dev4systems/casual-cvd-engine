import flwr as fl
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

# NOTE: For tree-based models like Random Forests, federated learning is complex.
# We demonstrate using a Deep Learning or Logistic Regression base for standard FL aggregation.

# --- CLIENT SCRIPT (Runs on individual hospital servers) ---
class HospitalClient(fl.client.NumPyClient):
    def __init__(self, model, X_train, y_train, X_test, y_test):
        self.model = model
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

    def get_parameters(self, config):
        # Extract weights from the local model
        return [self.model.coef_, self.model.intercept_]

    def set_parameters(self, parameters):
        # Update local model with globally aggregated weights
        self.model.coef_ = parameters[0]
        self.model.intercept_ = parameters[1]

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        # Train on local, private hospital data
        self.model.fit(self.X_train, self.y_train)
        return self.get_parameters(config=None), len(self.X_train), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        loss = log_loss(self.y_test, self.model.predict_proba(self.X_test))
        accuracy = self.model.score(self.X_test, self.y_test)
        return loss, len(self.X_test), {"accuracy": accuracy}

def start_client():
    # Initialize a local model and local data subset
    model = LogisticRegression(warm_start=True)
    model.classes_ = np.array([0, 1])
    # Assume X_train, y_train etc. are loaded here for Hospital A
    # client = HospitalClient(model, X_train, y_train, X_test, y_test)
    # fl.client.start_numpy_client(server_address="[::]:8080", client=client)
    pass


# --- SERVER SCRIPT (Runs on the central orchestrator) ---
def start_server():
    """
    Starts the central federated learning server to aggregate weights using Federated Averaging (FedAvg).
    """
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=1.0,           # Train on all available clients
        fraction_evaluate=1.0,      # Evaluate on all clients
        min_fit_clients=2,          # Require at least 2 hospitals to start
        min_evaluate_clients=2,
        min_available_clients=2
    )

    print("Starting Federated Learning Global Server...")
    fl.server.start_server(
        server_address="[::]:8080",
        config=fl.server.ServerConfig(num_rounds=3), # Run 3 rounds of federated training
        strategy=strategy
    )
