# ----------------------------------------------------
# mainOpti.py
# Author: Pascal Schirmer
# Date: 2025-10-14
# Description:
#   Physics-Informed Neural Network (PINN) using LSTM for
#   thermal modeling with automated hyperparameter optimization
#   via Optuna.
# ----------------------------------------------------

from src.opti import *


# ----------------------------------------------------
# Helper: safely initialize Optuna SQLite storage
# ----------------------------------------------------
def init_optuna_storage(base_path: str, db_name: str = "study.db"):
    """
    Ensures the Optuna database exists and has all tables.
    Returns a fully usable RDBStorage object.
    """
    mdl_path = os.path.join(base_path, "mdl")
    os.makedirs(mdl_path, exist_ok=True)
    db_path = os.path.abspath(os.path.join(mdl_path, db_name))
    storage_url = f"sqlite:///{db_path.replace(os.sep, '/')}"

    print(f"📂 Using Optuna storage at: {storage_url}")

    # Create RDBStorage (allow table creation)
    storage = optuna.storages.RDBStorage(
        url=storage_url,
        engine_kwargs={"connect_args": {"check_same_thread": False}},
        skip_compatibility_check=False,
    )

    # Initialize schema if missing by creating a dummy study
    try:
        _ = optuna.create_study(
            study_name="__init_study__",
            storage=storage,
            load_if_exists=True,
            direction="minimize",
        )
        # Delete temp study if you don’t want it to clutter DB
        optuna.delete_study(study_name="__init_study__", storage=storage_url)
        print("✅ Optuna DB schema verified/created.")
    except Exception as e:
        print("⚠️ Could not delete temp init study (not critical):", e)

    return storage_url


# ----------------------------------------------------
# Main execution
# ----------------------------------------------------
def main():
    print("🚀 Starting Optuna Hyperparameter Optimization for PINN LSTM...")

    # --- define your base project path here ---
    base_path = os.path.abspath(".")  # or an explicit path to your repo root
    storage_url = init_optuna_storage(base_path)

    # --- create the study using persistent storage ---
    study = optuna.create_study(
        study_name="thermal_pinn_opt",
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=42),
        storage=storage_url,
        load_if_exists=True,
    )

    # --- run optimization ---
    study.optimize(objective, n_trials=40, timeout=3 * 3600, gc_after_trial=True)

    print("\n✅ Optimization Completed")
    print("Best Parameters:")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")
    print(f"Best Validation Loss: {study.best_value:.6f}")

    # --- save results ---
    results_csv = os.path.join(base_path, "mdl", "optuna_results.csv")
    study.trials_dataframe().to_csv(results_csv, index=False)
    print(f"📊 Results saved to: {results_csv}")

    # --- visualize ---
    optuna.visualization.plot_optimization_history(study).show()
    optuna.visualization.plot_param_importances(study).show()

    print(f"\n🗂 You can now view the dashboard with:\n"
          f"   optuna-dashboard {storage_url}\n")


if __name__ == "__main__":
    main()
