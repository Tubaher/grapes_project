import pandas as pd
import glob
import argparse
import os

"""
    Este Módulo almacena todas las detecciones de los pickles de una carpeta en 
    un solo pickle de DataFrame. Se filtran las detecciones repetidas.
    Las columnas del DataFrame generado siguen la siguiente estructura:
    COLUMNAS = CAMPO_ID | CUARTEL_ID | HILERA_ID | AM/PM | RACIMO_ID | AREA_ID
"""

if __name__ == '__main__':
    # Parse command line arguments
    parser = argparse.ArgumentParser(
		description='Generate Mega Pickle')
        
    parser.add_argument('--pickles_dir', required=False,
                        default="stuff/pickles/TestCampo",
						metavar="/path/to/pickles_files.pkl",
						help="Path to pickles .pkl files")

    args = parser.parse_args()

    predictions_dir = os.path.join(args.pickles_dir,"prediction_pickles/")
    output_dir = args.pickles_dir

    #List to stack the dframes
    df_frames = []

    #Open each pickle file and append to a list
    for file in glob.glob(predictions_dir+"*.pkl"):
        
        #Read pickle and transform to a data frame
        df = pd.read_pickle(file)
        df = pd.DataFrame(df, columns = ["campo_id","cuartel_id","hilera_id","am/pm","racimo_id","racimo_area"])
        
        #Filter the repeated racimos, only chose the first ocurrence
        df = df.groupby(["racimo_id"]).first().reset_index()

        #Append to the list
        df_frames.append(df)

    #Create a data frame of all the pickles
    full_df = pd.concat(df_frames).reset_index()

    #Rename and store the new pickle
    df_output = full_df.rename(columns= {i : str(i).upper() for i in list(full_df.columns)})
    df_output = df_output[["CAMPO_ID","CUARTEL_ID","HILERA_ID","AM/PM","RACIMO_ID","RACIMO_AREA"]]

    print("Saving data frame to mega pickle...")
    output_predictions_dir = os.path.join(output_dir,"mega_pickle.pkl")
    df_output.to_pickle(output_predictions_dir)

    print(df_output.head())