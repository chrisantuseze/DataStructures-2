from dataclasses import dataclass, field

@dataclass(order=True)
class Query1Data:
    token_id: int
    n_txns: int

@dataclass(order=True)
class Query1Input:
    txn_hash: str
    token_id: int
