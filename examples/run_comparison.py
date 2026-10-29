"""Compare the two methods on one generated network."""
from two_layer_ctmc import build_two_layer_network, simulate_micro, simulate_micromacro


def main() -> None:
    network = build_two_layer_network(
        n_communities=3, community_size=10, inter_links=2,
        macro_graph_type="chain", seed=7,
    )
    common = dict(network=network, gamma=0.1, model="SIR", T_end=20.0,
                  initial_node=0, seed=123)
    full = simulate_micro(beta=0.2, dt_out=1.0, **common)
    times, infected, _, events = simulate_micromacro(
        beta_micro=0.2, tau_micro=0.5, **common
    )
    print(f"Full: {len(full['rows'])} community observations")
    print(f"TwoScale: {len(times)} times, {len(events)} events")
    print(f"TwoScale infected at its final observation: {infected[-1]}")


if __name__ == "__main__":
    main()
