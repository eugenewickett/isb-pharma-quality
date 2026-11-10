# Code for building and solving the WES network-flow problem
# 8-NOV-26

import os
import math
import pandas as pd
import pulp


# def generate_sample_excel(file_path: str = "networkflowFiles/input_data.xlsx"):
#     """
#     Generates a sample Excel workbook containing the 4 required parameter sheets.
#     """
#     os.makedirs(os.path.dirname(file_path), exist_ok=True)
#
#     # 1. City Demand Sheet (2 Cities)
#     city_demand_df = pd.DataFrame({
#         'City': ['New Delhi', 'Mumbai'],
#         'Outflow': [100, 150]
#     })
#
#     # 2. Lab Parameters Sheet (2 Labs)
#     lab_params_df = pd.DataFrame({
#         'Lab_A': [50, 20, 300, 1500],
#         'Lab_B': [200, 25, 350, 2000]
#     }, index=['Capacity', 'Batch Size', 'Testing Cost per Batch', 'Fixed Usage/Setup Cost'])
#
#     # 3. Per-Sample Variable Shipping Cost Matrix (c_ij)
#     cities = ['New Delhi', 'Mumbai']
#     labs = ['Lab_A', 'Lab_B']
#
#     per_sample_cost_df = pd.DataFrame([
#         [4, 8],
#         [7, 3]
#     ], index=cities, columns=labs)
#
#     # 4. Per-Route Fixed Activation Cost Matrix (r_ij)
#     per_route_fixed_cost_df = pd.DataFrame([
#         [500, 800],
#         [750, 400]
#     ], index=cities, columns=labs)
#
#     # Write to Excel workbook with multiple sheets
#     with pd.ExcelWriter(file_path) as writer:
#         city_demand_df.to_excel(writer, sheet_name='City_Demand', index=False)
#         lab_params_df.to_excel(writer, sheet_name='Lab_Parameters')
#         per_sample_cost_df.to_excel(writer, sheet_name='Per_Sample_Cost')
#         per_route_fixed_cost_df.to_excel(writer, sheet_name='Per_Route_Fixed_Cost')
#
#     print(f"✓ Sample dataset generated at: {file_path}")


def load_inputs(file_path: str = "networkflowFiles/input_data_big.xlsx"):
    """
    Reads parameters from the 15 Excel sheets and returns structured data dictionaries.
    """
    df_cities = pd.read_excel(file_path, sheet_name='City_Demand')
    df_r = pd.read_excel(file_path, sheet_name='Per_Route_Fixed_Cost', index_col=0)
    df_c_transport = pd.read_excel(file_path, sheet_name='Per_Sample_Transport_Cost', index_col=0)
    df_c_process = pd.read_excel(file_path, sheet_name='Per_Sample_Processing_Cost', index_col=0)
    df_lab_setup = pd.read_excel(file_path, sheet_name='Lab_Setup_Cost', index_col=0)
    df_lab_resource_avail = pd.read_excel(file_path, sheet_name='Lab_Resource_Available', index_col=0)
    df_lab_resource_purchase_cost = pd.read_excel(file_path, sheet_name='Lab_Resource_Purchase_Cost', index_col=0)
    df_lab_resource_fixed_cost = pd.read_excel(file_path, sheet_name='Lab_Resource_Fixed_Cost', index_col=0)
    df_lab_resource_capacity = pd.read_excel(file_path, sheet_name='Lab_Resource_Capacity', index_col=0)
    df_pathA_batch_size = pd.read_excel(file_path, sheet_name='PathogenA_Batch_Sizes', index_col=0)
    df_pathA_batch_cost = pd.read_excel(file_path, sheet_name='PathogenA_Batch_Costs', index_col=0)
    df_pathB_batch_size = pd.read_excel(file_path, sheet_name='PathogenB_Batch_Sizes', index_col=0)
    df_pathB_batch_cost = pd.read_excel(file_path, sheet_name='PathogenB_Batch_Costs', index_col=0)
    df_pathC_batch_size = pd.read_excel(file_path, sheet_name='PathogenC_Batch_Sizes', index_col=0)
    df_pathC_batch_cost = pd.read_excel(file_path, sheet_name='PathogenC_Batch_Costs', index_col=0)
    #df_labs = pd.read_excel(file_path, sheet_name='Lab_Parameters', index_col=0)
    #df_c = pd.read_excel(file_path, sheet_name='Per_Sample_Cost', index_col=0)


    # Get string indices/columns
    df_cities['City'] = df_cities['City'].astype(str).str.strip()
    df_r.index = df_r.index.astype(str).str.strip()
    df_r.columns = df_r.columns.astype(str).str.strip()
    df_c_transport.index = df_c_transport.index.astype(str).str.strip()
    df_c_transport.columns = df_c_transport.columns.astype(str).str.strip()
    df_c_process.index = df_c_process.index.astype(str).str.strip()
    df_c_process.columns = df_c_process.columns.astype(str).str.strip()
    df_lab_setup.index = df_lab_setup.index.astype(str).str.strip()
    df_lab_setup.columns = df_lab_setup.columns.astype(str).str.strip()
    df_lab_resource_avail.index = df_lab_resource_avail.index.astype(str).str.strip()
    df_lab_resource_avail.columns = df_lab_resource_avail.columns.astype(str).str.strip()
    df_lab_resource_purchase_cost.index = df_lab_resource_purchase_cost.index.astype(str).str.strip()
    df_lab_resource_purchase_cost.columns = df_lab_resource_purchase_cost.columns.astype(str).str.strip()
    df_lab_resource_fixed_cost.index = df_lab_resource_fixed_cost.index.astype(str).str.strip()
    df_lab_resource_fixed_cost.columns = df_lab_resource_fixed_cost.columns.astype(str).str.strip()
    df_lab_resource_capacity.index = df_lab_resource_capacity.index.astype(str).str.strip()
    df_lab_resource_capacity.columns = df_lab_resource_capacity.columns.astype(str).str.strip()
    df_pathA_batch_size.index = df_pathA_batch_size.index.astype(str).str.strip()
    df_pathA_batch_size.columns = df_pathA_batch_size.columns.astype(str).str.strip()
    df_pathA_batch_cost.index = df_pathA_batch_cost.index.astype(str).str.strip()
    df_pathA_batch_cost.columns = df_pathA_batch_cost.columns.astype(str).str.strip()
    df_pathB_batch_size.index = df_pathB_batch_size.index.astype(str).str.strip()
    df_pathB_batch_size.columns = df_pathB_batch_size.columns.astype(str).str.strip()
    df_pathB_batch_cost.index = df_pathB_batch_cost.index.astype(str).str.strip()
    df_pathB_batch_cost.columns = df_pathB_batch_cost.columns.astype(str).str.strip()
    df_pathC_batch_size.index = df_pathC_batch_size.index.astype(str).str.strip()
    df_pathC_batch_size.columns = df_pathC_batch_size.columns.astype(str).str.strip()
    df_pathC_batch_cost.index = df_pathC_batch_cost.index.astype(str).str.strip()
    df_pathC_batch_cost.columns = df_pathC_batch_cost.columns.astype(str).str.strip()

    # Get base sets (I, J, K, L, S)
    cities = df_cities['City'].tolist()
    labs = df_lab_setup.index.tolist()
    pathogens = df_cities.columns[1:].tolist()
    resources = df_lab_resource_avail.columns.tolist()
    steps = df_pathA_batch_size.columns.tolist()

    # Flow demands by pathogen (d_ik)
    demA = df_cities.set_index('City')['PathogenA'].to_dict()
    demB = df_cities.set_index('City')['PathogenB'].to_dict()
    demC = df_cities.set_index('City')['PathogenC'].to_dict()

    # Transportation parameters (r_ij, c_ij, c_k)
    route_fixed_costs = df_r.stack().to_dict()
    unit_transport_costs = df_c_transport.stack().to_dict()
    unit_process_costs = df_c_process['Cost'].to_dict()

    # Lab parameters (F_j, hat{u}_jl )
    lab_fixed_setup_costs = {}     # F_j
    for lab in labs:
        lab_fixed_setup_costs[lab] = float(df_lab_setup.loc[lab]['Fixed Usage/Setup Cost'])
    lab_resource_avail = df_lab_resource_avail.stack().to_dict()    # hat{u}_jl
    lab_resource_purchase_cost = df_lab_resource_purchase_cost.stack().to_dict()  # v_jl
    pathogen_resource_fixed_cost = df_lab_resource_fixed_cost.stack().to_dict() # f_kl, assumed same across labs
    pathogen_resource_capacity = df_lab_resource_capacity.stack().to_dict()  # m_kl, assumed same across labs
    pathogenA_batch_size = df_pathA_batch_size.stack().to_dict()  # b_ls, broken up by pathogen and assumed constant across labs
    pathogenA_batch_cost = df_pathA_batch_cost.stack().to_dict()  # g_ls, broken up by pathogen and assumed constant across labs
    pathogenB_batch_size = df_pathB_batch_size.stack().to_dict()  # b_ls, broken up by pathogen and assumed constant across labs
    pathogenB_batch_cost = df_pathB_batch_cost.stack().to_dict()  # g_ls, broken up by pathogen and assumed constant across labs
    pathogenC_batch_size = df_pathC_batch_size.stack().to_dict()  # b_ls, broken up by pathogen and assumed constant across labs
    pathogenC_batch_cost = df_pathC_batch_cost.stack().to_dict()  # g_ls, broken up by pathogen and assumed constant across labs

    #capacities = {}
    #batch_sizes = {}
    #batch_costs = {}

    #lab_dict = df_labs.to_dict()

    # capacities[lab] = float(lab_dict[lab]['Capacity'])
    # batch_sizes[lab] = float(lab_dict[lab]['Batch Size'])
    # batch_costs[lab] = float(lab_dict[lab]['Testing Cost per Batch'])



    #
    # input_dfs = {
    #     'cities': df_cities,
    #     'labs': df_labs,
    #     'per_sample_cost': df_c,
    #     'per_route_fixed_cost': df_r
    # }

    return cities, labs, pathogens, resources, steps, demA, demB, demC, route_fixed_costs, \
        unit_transport_costs, unit_process_costs, lab_fixed_setup_costs, lab_resource_avail, \
        lab_resource_purchase_cost, pathogen_resource_fixed_cost, pathogen_resource_capacity, \
        pathogenA_batch_size, pathogenA_batch_cost, pathogenB_batch_size, pathogenB_batch_cost, \
        pathogenC_batch_size, pathogenC_batch_cost


# def print_inputs(input_dfs: dict):
#     """
#     Prints all input parameter sheets clearly to the console.
#     """
#     print("\n=================== INPUT DATA ===================")
#     print("1. City Demand (d_i):")
#     print(input_dfs['cities'].to_string(index=False))
#
#     print("\n2. Lab Parameters:")
#     print(input_dfs['labs'].to_string())
#
#     print("\n3. Per-Sample Variable Shipping Cost (c_ij):")
#     print(input_dfs['per_sample_cost'].to_string())
#
#     print("\n4. Per-Route Fixed Activation Cost (r_ij):")
#     print(input_dfs['per_route_fixed_cost'].to_string())
#     print("==================================================\n")

# def validate_inputs(demands: dict, capacities: dict):
#     """
#     Checks problem feasibility: total capacity >= total demand.
#     """
#     total_demand = sum(demands.values())
#     total_capacity = sum(capacities.values())
#
#     if total_capacity < total_demand:
#         raise ValueError(
#             f"Infeasible Model: Total Lab Capacity ({total_capacity}) "
#             f"is less than Total City Outflow/Demand ({total_demand})."
#         )
#
#     print(f"✓ Ingestion Validation Passed: Demand ({total_demand}) <= Capacity ({total_capacity})")



def solve_cost_optimization(file_path: str = "networkflowFiles/input_data_big.xlsx", verbose: bool = True,
                            timeLimit: float = 500, relativeGap: float = 0.01):
    """
    Constructs and solves the sample routing and lab optimization model.
    """
    # Load parameters from excel file
    cities, labs, pathogens, resources, steps, demA, demB, demC, route_fixed_costs, unit_transport_costs, \
        unit_process_costs, lab_fixed_setup_costs, lab_resource_avail, \
        lab_resource_purchase_cost, pathogen_resource_fixed_cost, pathogen_resource_capacity, \
        pathogenA_batch_size, pathogenA_batch_cost, pathogenB_batch_size, pathogenB_batch_cost, \
        pathogenC_batch_size, pathogenC_batch_cost = load_inputs(file_path)
    # print_inputs(input_dfs)
    # validate_inputs(demands, capacities)

    # 2. Dynamic tight Big-M bounds calculation  # TODO: Figure something out for this later
    # M_route = {
    #     (i, j): min(demands[i], capacities[j])
    #     for i in cities for j in labs
    # }
    # M_lab = {
    #     j: math.ceil(capacities[j] / batch_sizes[j])
    #     for j in labs
    # }

    bigM = 10000

    # 3. Initialize LP Problem
    model = pulp.LpProblem("WES_Network_Flow_Optimization", pulp.LpMinimize)

    # 4. Decision Variables
    x = pulp.LpVariable.dicts("x", (cities, labs, pathogens), lowBound=0, cat=pulp.LpInteger)
    y = pulp.LpVariable.dicts("y", (cities, labs), cat=pulp.LpBinary)
    W = pulp.LpVariable.dicts("W", labs, cat=pulp.LpBinary)
    w = pulp.LpVariable.dicts("w", (labs, pathogens, resources), cat=pulp.LpBinary)
    u = pulp.LpVariable.dicts("u", (labs, resources), lowBound=0, cat=pulp.LpInteger)
    z = pulp.LpVariable.dicts("z", (labs, pathogens, resources, steps), lowBound=0, cat=pulp.LpInteger)

    # 5. Objective Function
    route_costs = pulp.lpSum(
        route_fixed_costs[i, j] * y[i][j] + unit_transport_costs[i, j] * pulp.lpSum(x[i][j][k] for k in pathogens)
        for i in cities for j in labs
    )
    process_costs = pulp.lpSum(
        unit_process_costs[k] * pulp.lpSum(x[i][j][k] for i in cities for j in labs)
        for k in pathogens
    )
    new_lab_costs = pulp.lpSum(
        lab_fixed_setup_costs[j] * W[j]
        for j in labs
    )
    new_resource_costs = pulp.lpSum(
        lab_resource_purchase_cost[j, l] * u[j][l]
        for j in labs for l in resources
    )
    lab_equip_costs = pulp.lpSum(
        pathogen_resource_fixed_cost[k, l] * w[j][k][l]
        for j in labs for k in pathogens for l in resources
    )
    lab_batch_processing_costs = pulp.lpSum(
        pathogenA_batch_cost[l, s] * z[j]['PathogenA'][l][s] + pathogenB_batch_cost[l, s] * z[j]['PathogenB'][l][s] +
        pathogenC_batch_cost[l, s] * z[j]['PathogenC'][l][s]
        for j in labs for l in resources for s in steps
    )
    model += route_costs + process_costs + new_lab_costs + new_resource_costs + lab_equip_costs + lab_batch_processing_costs, "Total_Operational_Cost"

    # 6. Constraints
    # (1b) Demand is met
    for i in cities:
        model += (
            pulp.lpSum(x[i][j]['PathogenA'] for j in labs) == demA[i],
            f"Demand_Satisfaction_{i}_{'PathogenA'}"
        )
        model += (
            pulp.lpSum(x[i][j]['PathogenB'] for j in labs) == demB[i],
            f"Demand_Satisfaction_{i}_{'PathogenB'}"
        )
        model += (
            pulp.lpSum(x[i][j]['PathogenC'] for j in labs) == demC[i],
            f"Demand_Satisfaction_{i}_{'PathogenC'}"
        )

    # (1c) Route activation
    for i in cities:
        for j in labs: # TODO: Replace with tighter Big M later
            model += (
                pulp.lpSum(x[i][j][k] for k in pathogens) <= bigM * y[i][j],
                f"Route_Activation_{i}_{j}"
            )

    # (1d) Lab activation
    for j in labs:
        model += (
            pulp.lpSum(x[i][j][k] for i in cities for k in pathogens) <= bigM * W[j],
            f"Lab_Activation_{j}"
        )

    # (1e) Lab resource activation
    for j in labs:
        for k in pathogens:
            for l in resources:
                model += (
                    pulp.lpSum(x[i][j][k] for i in cities) <= bigM * w[j][k][l],
                    f"Lab_Activation_{j}_{k}_{l}"
                )

    # (1f) Samples are in a batch
    for j in labs:
        for l in resources:
            for s in steps:
                model += (
                    pulp.lpSum(x[i][j]['PathogenA'] for i in cities) <= pathogenA_batch_size[l, s] * z[j][k][l][s],
                    f"Sample_Batch_Inclusion_{j}_{'PathogenA'}_{l}_{s}"
                )
                model += (
                    pulp.lpSum(x[i][j]['PathogenB'] for i in cities) <= pathogenB_batch_size[l, s] * z[j][k][l][s],
                    f"Sample_Batch_Inclusion_{j}_{'PathogenB'}_{l}_{s}"
                )
                model += (
                    pulp.lpSum(x[i][j]['PathogenC'] for i in cities) <= pathogenC_batch_size[l, s] * z[j][k][l][s],
                    f"Sample_Batch_Inclusion_{j}_{'PathogenC'}_{l}_{s}"
                )

    # (1g) Resource capacity
    for j in labs:
        for l in resources:
            model += (
                pulp.lpSum( 1/pathogen_resource_capacity[k, l] * pulp.lpSum(x[i][j][k] for i in cities) for k in pathogens) <= lab_resource_avail[j, l] + u[j][l],
                f"Resource_Capacity_{j}_{l}"
            )

    # 7. Solve
    # solver = pulp.PULP_CBC_CMD(msg='verbose')
    # cbc_options = [
    #     "presolve more",  # More model simplification
    #     # "-pump 1",  # Use Feasibility Pump to quickly find better solutions
    #     # "-heuristics on",  # Enable all primal heuristics
    #     # "-cuts on",  # More cutting planes
    #     # "-passC 40"  # More iterations generating root-node cuts
    # ]

    # solver = pulp.PULP_CBC_CMD(msg=True,
    #                             timeLimit=400,
    #                             gapRel=0.05,
    #                             options=cbc_options
    # )
    # log_file = "highs_solver.log"
    solver = pulp.HiGHS(msg=verbose,
                        timeLimit=timeLimit,
                        gapRel=relativeGap,
                        #logPath=log_file,
    )
    print("\n Solving the optimization model: ")
    print(model)
    print("\n")

    status_code = model.solve(solver)
    total_cost = pulp.value(model.objective)

    # Breakdown of new resources
    new_resource_list = []
    for j in labs:
        for l in resources:
            if u[j][l].varValue > 0.8:
                new_resource_list.append({
                    'Lab': j,
                    'Resource': l,
                    'Number procured': int(u[j][l].varValue)
                })
    new_resource_df = pd.DataFrame(new_resource_list)

    # Gather active routes
    route_list = []
    for i in cities:
        for j in labs:
            if y[i][j].varValue == 1:
                for k in pathogens:
                    route_list.append({
                        'City': i,
                        'Lab': j,
                        'Pathogen': k,
                        'Sample Flow': int(x[i][j][k].varValue)
                    })
    routes_df = pd.DataFrame(route_list)

    # Summarize number of labs used by each city
    route_summary_list = []
    for i in cities:
        numLabs = 0
        for j in labs:
            if y[i][j].varValue == 1:
                numLabs += 1
        route_summary_list.append({
            'City': i,
            'Labs Used': numLabs,
        })
    routes_summary_df = pd.DataFrame(route_summary_list)

    # Where new labs are set up
    new_lab_list = []
    for j in labs:
        if W[j].varValue == 1 and lab_fixed_setup_costs[j] > 0.0:
            new_lab_list.append({
                'Lab': j
            })
    new_lab_df = pd.DataFrame(new_lab_list)

    # Gather which pathogens each lab is equipped for
    lab_pathgoen_equipped_list = []
    for j in labs:
        tempList = []
        for k in pathogens:
            equipped_Bool = False
            for l in resources:
                if w[j][k][l].varValue == 1:
                    equipped_Bool = True
            if equipped_Bool == True:
                tempList.append('Yes')
            elif equipped_Bool == False:
                tempList.append('No')
        lab_pathgoen_equipped_list.append({
            'Lab': j,
            pathogens[0]: tempList[0],
            pathogens[1]: tempList[1],
            pathogens[2]: tempList[2]
        })
    lab_pathgoen_equipped_df = pd.DataFrame(lab_pathgoen_equipped_list)
    # Breakdown of costs in objective
    soln_cost_df = pd.DataFrame([
        {'Category': 'Route costs', 'Amount': route_costs.value()},
        {'Category': 'Processing costs', 'Amount': process_costs.value()},
        {'Category': 'New lab costs', 'Amount': new_lab_costs.value()},
        {'Category': 'New resource costs', 'Amount': new_resource_costs.value()},
        {'Category': 'Lab pathogen-equipping fixed costs', 'Amount': lab_equip_costs.value()},
        {'Category': 'Lab batch-processing costs', 'Amount': lab_batch_processing_costs.value()},
        {'Category': 'TOTAL MINIMIZED COST', 'Amount': total_cost}
    ])

    return soln_cost_df, routes_df, routes_summary_df, new_resource_df, new_lab_df, lab_pathgoen_equipped_df


def export_to_excel(soln_cost_df: pd.DataFrame,
                    routes_df: pd.DataFrame,
                    routes_summary_df: pd.DataFrame,
                    new_resource_df: pd.DataFrame,
                    new_lab_df: pd.DataFrame,
                    lab_pathgoen_equipped_df: pd.DataFrame,
                    output_path: str = "networkflowFiles/opt_output.xlsx"):
    """
    Exports the optimization solution tables to an Excel workbook with multiple sheets.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with pd.ExcelWriter(output_path) as writer:
        soln_cost_df.to_excel(writer, sheet_name='Cost_Summary', index=False)
        routes_df.to_excel(writer, sheet_name='Route_Allocation', index=False)
        routes_summary_df.to_excel(writer, sheet_name='Route_Allocation_Summary', index=False)
        new_resource_df.to_excel(writer, sheet_name='Resources_Procured', index=False)
        new_lab_df.to_excel(writer, sheet_name='New_Labs', index=False)
        lab_pathgoen_equipped_df.to_excel(writer, sheet_name='Lab_Pathogen_Equipped', index=False)
    print(f"✓ Results successfully saved to: {output_path}")


# def main():
#     input_file = "networkflowFiles/input_data_big.xlsx"
#     output_file = "networkflowFiles/optimization_data.xlsx"
#
#     # 1. Generate sample data if not present
#     if not os.path.exists(input_file):
#         generate_sample_excel(input_file)
#
#     print("\n==================================================")
#     print(" RUNNING SAMPLE ROUTING & LAB OPTIMIZATION")
#     print("==================================================")
#
#     # 2. Run solver
#     status_str, total_cost, flow_df, lab_df, cost_df = solve_cost_optimization(
#         file_path=input_file,
#         verbose=True
#     )
#
#     # 3. Print Output Decomposition
#     print(f"Status: {status_str}")
#     print(f"Total Operational Cost: ₹{total_cost:,.2f}\n")
#     print("--- COST DECOMPOSITION ---")
#     print(cost_df.to_string(index=False))
#
#     # 4. Save results to Excel
#     export_to_excel(flow_df, lab_df, cost_df, output_file)


# if __name__ == "__main__":
#     main()

soln_cost_df, routes_df, routes_summary_df, new_resource_df, new_lab_df, \
    lab_pathgoen_equipped_df = solve_cost_optimization("networkflowFiles/input_data_big.xlsx", verbose=True,
                            timeLimit=50, relativeGap=0.05)

export_to_excel(soln_cost_df, routes_df, routes_summary_df, new_resource_df, new_lab_df, lab_pathgoen_equipped_df,
                output_path="networkflowFiles/opt_output.xlsx")

