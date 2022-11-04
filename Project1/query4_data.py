from dataclasses import dataclass

@dataclass(order=True)
class Query4Data:
    token_id: str
    # buyer: str
    n_unique_buyers: int


@dataclass(order=True)
class Query4Input:
    txn_hash: str
    token_id: int
    buyer: str