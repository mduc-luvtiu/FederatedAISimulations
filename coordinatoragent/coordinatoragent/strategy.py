# coordinatoragent/strategy.py

from typing import Dict, List, Optional, Tuple
import random
from flwr.common import Parameters, FitIns, Scalar
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg


class CoordinatorAgentStrategy(FedAvg):
    """Coordinator Agent Strategy: 
    - Observe client stats
    - Score and select clients
    - Send round-specific configs
    """

    def __init__(
        self,
        fraction_fit: float = 0.3,
        fraction_evaluate: float = 1.0,
        min_available_clients: int = 2,
        initial_parameters: Optional[Parameters] = None,
    ) -> None:
        super().__init__(
            fraction_fit=fraction_fit,
            fraction_evaluate=fraction_evaluate,
            min_available_clients=min_available_clients,
            initial_parameters=initial_parameters,
        )

        # Store client reliability scores (mock init = random)
        self.client_scores: Dict[str, float] = {}

    # -------- AGENT BEHAVIOR --------
    def score_clients(self, clients: List[ClientProxy]) -> Dict[str, float]:
        """Assign a reliability score to each client (simulate observation)."""
        scores = {}
        for c in clients:
            if c.cid not in self.client_scores:
                self.client_scores[c.cid] = random.uniform(0.5, 1.0)

            # Update nhẹ theo thời gian (giả lập học hỏi từ feedback)
            self.client_scores[c.cid] *= random.uniform(0.95, 1.05)

            scores[c.cid] = self.client_scores[c.cid]
        return scores

    # -------- OVERRIDES --------
    def configure_fit(
        self,
        server_round: int,
        parameters: Parameters,
        client_manager,
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """Coordinator Agent selects clients based on scores + sends config."""

        # Lấy tất cả client khả dụng
        available_clients = list(client_manager.all().values())
        scores = self.score_clients(available_clients)

        # Sắp xếp client theo score giảm dần
        sorted_clients = sorted(
            available_clients, key=lambda c: scores[c.cid], reverse=True
        )

        # Lấy top-K theo fraction_fit
        sample_size, _ = self.num_fit_clients(len(sorted_clients))
        selected_clients = sorted_clients[:sample_size]

        # Tạo config động cho vòng này
        config: Dict[str, Scalar] = {
            "server_round": server_round,
            "local_epochs": 1 if server_round < 3 else 2,
            "learning_rate": 0.01 if server_round < 5 else 0.005,
        }

        fit_ins = FitIns(parameters, config)

        return [(client, fit_ins) for client in selected_clients]

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, object]],
        failures,
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate updates + log participation."""
        aggregated_parameters, metrics = super().aggregate_fit(
            server_round, results, failures
        )

        # Log thêm: số client tham gia, trung bình score
        metrics["num_clients"] = len(results)
        if results:
            avg_score = sum(
                [self.client_scores[c.cid] for c, _ in results]
            ) / len(results)
            metrics["avg_client_score"] = avg_score

        return aggregated_parameters, metrics
