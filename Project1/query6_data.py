from dataclasses import dataclass, field

@dataclass(order=True)
class Query6Data:
    buyer: str
    nft: str
    n_txns_for_nft: int
    total_unique_nft: int
    total_txns: int

@dataclass(order=True)
class Query6Input:
    buyer: str
    nft: str
    token_id: int