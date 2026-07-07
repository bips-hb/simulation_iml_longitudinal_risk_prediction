from generate_dashboard import generate_multi_dashboard

generate_multi_dashboard(
    scenarios=[
        # eICU
        ('eICU In-Hospital Mortality L2', '/home/hoepler/eICU_Real_World_Example/model_results/mortality_L2/'),
        ('eICU AKI L2', '/home/hoepler/eICU_Real_World_Example/model_results/aki_L2/'),
        # Simulation — baseline
        ('Baseline', '/home/hoepler/eICU_Real_World_Example/simulation_results_advanced/baseline/'),
        # Simulation — non-baseline
        ('High Noise',    '/home/hoepler/eICU_Real_World_Example/simulation_results_advanced/high_noise/'),
        ('High Correlation',    '/home/hoepler/eICU_Real_World_Example/simulation_results_advanced/high_correlation/'),
        ('Short History',       '/home/hoepler/eICU_Real_World_Example/simulation_results_advanced/short_history/'),
    ],
    dashboard_path='/home/hoepler/eICU_Real_World_Example/dashboard_all.html',
)