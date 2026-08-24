"""The evaluation world: a synthetic company, its documents, and its questions.

Both the corpus and the ground-truth answers are generated from one seeded data
model, so every answer is correct *by construction*. Hand-writing questions over
a real corpus gets the answer wrong often enough to poison a fitness function,
and a poisoned fitness function is worse than none.

The world is fictional on purpose. A network answering questions about a real
company could be recalling its pretraining instead of retrieving, and then
fitness would be measuring the model rather than the topology. Nothing here
exists, so the only way to answer is to go and look.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

SEED = 20260821

CITIES = ["Halberd", "Kestrel", "Lowmoor", "Ashford", "Brindle", "Cartwright",
          "Denmark Hill", "Eastgate", "Fenwick", "Garrowby", "Hexham", "Ilkley",
          "Jarrow", "Knaresborough", "Ludlow", "Malton", "Northallerton",
          "Ossett", "Pickering", "Quorn", "Ripon", "Selby", "Tadcaster", "Ulverston"]
SURNAMES = ["Okonkwo", "Halvorsen", "Ramaswamy", "Delacroix", "Bergström",
            "Nakamura", "Oyelaran", "Vasquez", "Thornbury", "Achterberg",
            "Mbeki", "Lindqvist", "Petrossian", "Duarte", "Whitlock",
            "Sørensen", "Ibarra", "Fairweather", "Kowalczyk", "Aduba",
            "Renshaw", "Molnár", "Castellanos", "Birtwistle"]
GOODS = ["refrigerated pharmaceuticals", "bulk grain", "automotive parts",
         "flat-pack furniture", "industrial chemicals", "textiles",
         "consumer electronics", "brewing supplies", "paper stock", "glassware"]


@dataclass(frozen=True)
class Depot:
    code: str
    city: str
    manager: str
    bays: int
    opened: int


@dataclass(frozen=True)
class Contract:
    ref: str
    client: str
    depot_code: str
    goods: str
    penalty_per_hour: int
    annual_value: int


@dataclass(frozen=True)
class Incident:
    ref: str
    contract_ref: str
    hours_late: int
    year: int
    cause: str


@dataclass
class World:
    depots: list[Depot] = field(default_factory=list)
    contracts: list[Contract] = field(default_factory=list)
    incidents: list[Incident] = field(default_factory=list)

    def depot(self, code: str) -> Depot:
        return next(d for d in self.depots if d.code == code)

    def contract(self, ref: str) -> Contract:
        return next(c for c in self.contracts if c.ref == ref)


def build_world(seed: int = SEED) -> World:
    """Deterministic. The same seed always yields the same company."""
    rng = random.Random(seed)
    world = World()

    for index, city in enumerate(CITIES):
        world.depots.append(Depot(
            code=f"D{index + 1:02d}",
            city=city,
            manager=f"{rng.choice(['A.', 'J.', 'M.', 'R.', 'S.'])} {SURNAMES[index]}",
            bays=rng.choice([4, 6, 8, 12, 16]),
            opened=rng.randint(1998, 2019),
        ))

    clients = ["Verity Health", "Northwind Mills", "Calder Motors", "Studio Flatpack",
               "Peregrine Chemicals", "Loomcraft", "Bright Circuit", "Anvil Brewing",
               "Quire Paper", "Lucent Glass", "Harrowgate Foods", "Sable Freight",
               "Orchard Cold Chain", "Ironbridge Steel", "Marchmont Labs",
               "Copperfield Print", "Vantage Optics", "Redgate Textiles",
               "Silverbrook Dairy", "Thornbury Ceramics", "Ellesmere Pumps",
               "Brackenhill Seeds", "Cobalt Instruments", "Dunmore Packaging",
               "Fairlight Media", "Greystone Cement", "Havelock Timber",
               "Inverleith Spirits", "Juniper Cosmetics", "Kelvin Aerospace",
               "Larkspur Foods", "Mornington Tools", "Netherby Glass",
               "Ovingham Plastics", "Pentland Fibres", "Rossendale Wool",
               "Stourbridge Tiles", "Tallentire Feed", "Urswick Marine",
               "Wetheral Joinery"]
    for index, client in enumerate(clients):
        world.contracts.append(Contract(
            ref=f"C-{2100 + index}",
            client=client,
            depot_code=rng.choice(world.depots).code,
            goods=GOODS[index % len(GOODS)],
            # Distinct penalties so "highest" is never a tie.
            penalty_per_hour=250 + index * 125,
            annual_value=rng.randrange(400, 4000) * 1000,
        ))

    causes = ["driver shortage", "a failed refrigeration unit", "a bridge closure",
              "customs delay", "a mis-picked pallet", "storm damage"]
    for index in range(60):
        contract = world.contracts[index % len(world.contracts)]
        world.incidents.append(Incident(
            ref=f"INC-{4400 + index}",
            contract_ref=contract.ref,
            hours_late=rng.randint(2, 40),
            year=rng.choice([2024, 2025, 2026]),
            cause=rng.choice(causes),
        ))

    return world


def documents(world: World) -> dict[str, str]:
    """One document per fact-bearing entity. Deliberately spread across files so
    that a question spanning two entities cannot be answered from one document."""
    docs: dict[str, str] = {}

    for depot in world.depots:
        docs[f"depot-{depot.code}.txt"] = (
            f"Meridian Logistics depot {depot.code}\n"
            f"Location: {depot.city}\n"
            f"Depot manager: {depot.manager}\n"
            f"Loading bays: {depot.bays}\n"
            f"Opened: {depot.opened}\n"
        )

    for contract in world.contracts:
        docs[f"contract-{contract.ref}.txt"] = (
            f"Contract {contract.ref}\n"
            f"Client: {contract.client}\n"
            f"Serviced by depot: {contract.depot_code}\n"
            f"Goods: {contract.goods}\n"
            f"Late-delivery penalty: {contract.penalty_per_hour} per hour\n"
            f"Annual value: {contract.annual_value}\n"
        )

    for incident in world.incidents:
        docs[f"incident-{incident.ref}.txt"] = (
            f"Incident {incident.ref} ({incident.year})\n"
            f"Affected contract: {incident.contract_ref}\n"
            f"Hours late: {incident.hours_late}\n"
            f"Cause: {incident.cause}\n"
        )

    return docs
