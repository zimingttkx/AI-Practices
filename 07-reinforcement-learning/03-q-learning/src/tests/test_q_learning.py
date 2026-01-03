"""Unit Tests for Q-Learning Module.

Comprehensive test suite covering:
    - Agent implementations (Q-Learning, SARSA, Expected SARSA, Double Q)
    - Environment behavior (CliffWalking)
    - Training infrastructure
    - Exploration strategies
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import tempfile


def test_q_learning_update():
    """Test Q-Learning update correctness."""
    from agents import QLearningAgent

    agent = QLearningAgent(n_actions=4, learning_rate=0.5, discount_factor=0.9)
    state = (0, 0)
    next_state = (0, 1)

    # Set next state Q-values
    agent.q_table[next_state] = np.array([1.0, 2.0, 0.5, 0.3], dtype=np.float32)

    # Update: Q = 0 + 0.5 * (-1 + 0.9 * 2.0 - 0) = 0.4
    td_error = agent.update(state, 0, -1.0, next_state, False)

    expected_q = 0.5 * (-1.0 + 0.9 * 2.0)
    assert np.isclose(agent.q_table[state][0], expected_q, atol=1e-6), \
        f"Q-Learning update incorrect: {agent.q_table[state][0]} != {expected_q}"

    print("✓ Q-Learning update test passed")


def test_sarsa_update():
    """Test SARSA update correctness."""
    from agents import SARSAAgent

    agent = SARSAAgent(n_actions=4, learning_rate=0.5, discount_factor=0.9)
    state = (0, 0)
    next_state = (0, 1)

    agent.q_table[next_state] = np.array([1.0, 2.0, 0.5, 0.3], dtype=np.float32)

    # SARSA uses actual next action (1 in this case)
    td_error = agent.update(state, 0, -1.0, next_state, 1, False)

    expected_q = 0.5 * (-1.0 + 0.9 * 2.0)  # Q[next_state][1] = 2.0
    assert np.isclose(agent.q_table[state][0], expected_q, atol=1e-6), \
        f"SARSA update incorrect: {agent.q_table[state][0]} != {expected_q}"

    print("✓ SARSA update test passed")


def test_expected_sarsa():
    """Test Expected SARSA expected value calculation."""
    from agents import ExpectedSARSAAgent

    agent = ExpectedSARSAAgent(n_actions=4, epsilon=0.2)
    state = (0, 0)
    agent.q_table[state] = np.array([1.0, 2.0, 0.5, 0.5], dtype=np.float32)

    expected_q = agent._get_expected_q(state)

    # ε=0.2: best action (1) has prob 0.8 + 0.05 = 0.85, others 0.05 each
    manual_expected = 0.05 * 1.0 + 0.85 * 2.0 + 0.05 * 0.5 + 0.05 * 0.5

    assert np.isclose(expected_q, manual_expected, atol=1e-6), \
        f"Expected SARSA expectation incorrect: {expected_q} != {manual_expected}"

    print("✓ Expected SARSA test passed")


def test_double_q_learning():
    """Test Double Q-Learning updates both tables."""
    from agents import DoubleQLearningAgent

    agent = DoubleQLearningAgent(n_actions=4, learning_rate=0.5)
    state = (0, 0)
    next_state = (0, 1)

    np.random.seed(42)
    for _ in range(20):
        agent.update(state, 0, -1.0, next_state, False)

    q1_updated = agent.q_table[state][0] != 0
    q2_updated = agent.q_table2[state][0] != 0

    assert q1_updated or q2_updated, "At least one Q-table should be updated"

    print("✓ Double Q-Learning test passed")


def test_epsilon_greedy():
    """Test ε-greedy action selection."""
    from agents import QLearningAgent

    # Test greedy (ε=0)
    agent = QLearningAgent(n_actions=4, epsilon=0.0)
    state = (0, 0)
    agent.q_table[state] = np.array([1.0, 3.0, 2.0, 0.5], dtype=np.float32)

    actions = [agent.get_action(state, training=True) for _ in range(100)]
    assert all(a == 1 for a in actions), "ε=0 should always pick best action"

    # Test exploration (ε>0)
    agent.epsilon = 0.5
    np.random.seed(42)
    actions = [agent.get_action(state, training=True) for _ in range(1000)]
    unique_actions = set(actions)
    assert len(unique_actions) > 1, "ε=0.5 should explore multiple actions"

    print("✓ ε-greedy test passed")


def test_cliff_walking_env():
    """Test CliffWalking environment."""
    from environments import CliffWalkingEnv

    env = CliffWalkingEnv()

    # Test reset
    state = env.reset()
    assert state == (3, 0), f"Start state incorrect: {state}"

    # Test move up
    next_state, reward, done = env.step(0)
    assert next_state == (2, 0), f"Move up incorrect: {next_state}"
    assert reward == -1.0, f"Step reward incorrect: {reward}"
    assert not done, "Should not terminate"

    # Test cliff
    env.reset()
    next_state, reward, done = env.step(1)  # Right into cliff
    assert reward == -100.0, f"Cliff reward incorrect: {reward}"
    assert next_state == env.start, "Should reset to start after cliff"

    # Test goal
    env.reset()
    env.state = (3, 10)
    next_state, reward, done = env.step(1)  # Right to goal
    assert next_state == env.goal, f"Should reach goal: {next_state}"
    assert done, "Should terminate at goal"

    print("✓ CliffWalking environment test passed")


def test_save_load():
    """Test agent save/load functionality."""
    from agents import QLearningAgent

    agent = QLearningAgent(n_actions=4)
    agent.q_table[(0, 0)] = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        filepath = f.name

    agent.save(filepath)

    new_agent = QLearningAgent(n_actions=4)
    new_agent.load(filepath)

    assert np.allclose(new_agent.q_table[(0, 0)], agent.q_table[(0, 0)]), \
        "Loaded Q-table doesn't match"

    Path(filepath).unlink()

    print("✓ Save/load test passed")


def test_training_convergence():
    """Test that Q-Learning converges on CliffWalking."""
    from agents import QLearningAgent
    from environments import CliffWalkingEnv
    from training import train_q_learning

    env = CliffWalkingEnv()
    agent = QLearningAgent(
        n_actions=4,
        learning_rate=0.5,
        epsilon=0.1,
        epsilon_decay=1.0,
        epsilon_min=0.1,
    )

    metrics = train_q_learning(env, agent, episodes=300, verbose=False)

    final_avg = np.mean(metrics.episode_rewards[-50:])
    assert final_avg > -100, f"Training didn't converge: avg={final_avg}"

    print(f"✓ Training convergence test passed (final avg: {final_avg:.2f})")


def test_exploration_strategies():
    """Test exploration strategy implementations."""
    from exploration import ExplorationMixin

    mixin = ExplorationMixin()
    q_vals = np.array([1.0, 3.0, 2.0, 0.5])

    # ε-greedy with ε=0 should be greedy
    actions = [mixin._epsilon_greedy(q_vals, 0.0) for _ in range(100)]
    assert all(a == 1 for a in actions), "ε=0 should be greedy"

    # Softmax temperature effect
    np.random.seed(42)
    cold_actions = [mixin._softmax(q_vals, 0.1) for _ in range(1000)]
    hot_actions = [mixin._softmax(q_vals, 10.0) for _ in range(1000)]

    cold_best_freq = cold_actions.count(1) / 1000
    hot_best_freq = hot_actions.count(1) / 1000
    assert cold_best_freq > hot_best_freq, "Low τ should be more greedy"

    # UCB prioritizes unvisited
    counts = np.array([10, 0, 5, 3])
    action = mixin._ucb(q_vals, counts, 18, c=2.0)
    assert action == 1, "UCB should pick unvisited action"

    print("✓ Exploration strategies test passed")


def run_all_tests():
    """Run all unit tests."""
    print("=" * 60)
    print("Running Q-Learning Module Unit Tests")
    print("=" * 60)
    print()

    tests = [
        test_q_learning_update,
        test_sarsa_update,
        test_expected_sarsa,
        test_double_q_learning,
        test_epsilon_greedy,
        test_cliff_walking_env,
        test_save_load,
        test_training_convergence,
        test_exploration_strategies,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"✗ {test.__name__} FAILED: {e}")
            failed += 1

    print()
    print("=" * 60)
    print(f"Tests completed: {passed} passed, {failed} failed")

    if failed == 0:
        print("All tests passed!")
    else:
        print("Some tests failed. Please check the output above.")
    print("=" * 60)

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
