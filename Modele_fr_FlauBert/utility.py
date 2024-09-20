import pandas as pd
import plotly.graph_objects as go


def nb_exemple_categorie(df):
    count_exemple_df = pd.DataFrame(columns=["Label", "Nombre_de_exemples"])


    for label in df.columns:
        count = df[df[label] == 1].shape[0]  
        count_exemple_df = count_exemple_df.append({"Label": label, "Nombre_de_exemples": count}, ignore_index=True)


    return count_exemple_df


def plot_distribution(df,titre="distribution du nombre de document par catégorie"):
    
    nb_labele=nb_exemple_categorie(df)
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=nb_labele["Label"],
        y=nb_labele["Nombre_de_exemples"],
        marker_color='rgb(55, 83, 109)'
    ))

    fig.update_layout(
        title=titre,
        xaxis_title="Catégories",
        yaxis_title="Nombre de documents",
        xaxis_tickangle=-25,
        font=dict(
            family="Arial, sans-serif",
            size=13,
            color="RebeccaPurple"
        )
    )

    fig.show()


def Liste_categories():
    
    categories=["building_works","studies_consulting_assistance","transport_and_related_services",
"financial_and_insurance_services","sewerage","car_industry_products_and_transport_equipment",
"real_estate_missions_and_inspections","it_services","cleaning_and_hygiene_services",
"building_equipment","medical_precision_optical_and_watchmaking_instruments",
"research_and_development","civil_engineering","green_spaces",
"printing_and_related_services","street_development","office_and_it_equipment",
"industrial_equipment_and_tools","electrical_equipment","chemical_products",
"structural_works","recreational_cultural_and_sports_services","electricity_gas_and_heat",
"food_beverages_and_related_products","health_and_social_work_services",
"education_and_training_services","radio_television_and_communication_equipment",
"clothing_and_textile_industry","roofing","communication_and_marketing_services",
"energy_and_environment","waste_management","ores_materials_and_construction_materials",
"software_supply","defence_and_security","petroleum_products_fuels",
"office_furniture_and_supplies","it_equipment_and_consumables","provision_of_meals",
"fire","postal_and_telecommunications_services","silvicultural_products",
"drinking_water","agriculture","wholesale_and_retail","water_related_works",
"mining_and_minerals","paper_and_cardboard","recreational_cultural_and_sports_equipment",
"rubber_or_plastic_products","hydraulic_equipment"]
    
    

    return categories


