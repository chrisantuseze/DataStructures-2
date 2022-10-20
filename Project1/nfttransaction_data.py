from dataclasses import dataclass, field

@dataclass(order=True)
class NFTTransaction:
    txn_hash: str
    time_stamp: str
    date_time: str
    action: str
    buyer: str
    nft: str
    token_id: int
    type_: int
    quantity: int
    price: str
    market: str
    n_unique_buyers: int
