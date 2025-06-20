import os
import pandas as pd
import numpy as np


def generate_participants(main_dir):
    data_path = os.path.join(main_dir, 'sourcedata/all_patients_all_dates_depersonalised')
    patients = os.listdir(data_path)

    patients_id = ["sub-{}".format('%03d' % (idx + 1)) for idx in range(0, len(patients))]
    df_participants = pd.DataFrame(patients, columns=['Patient'], index=np.arange(0, len(patients_id)))
    df_participants['Sub-ID'] = patients_id
    df_participants['Age'] = None
    df_participants['Sex'] = None
    df_participants.to_csv(main_dir + '/participants.tsv', sep="\t")


if __name__ == '__main__':
    cwd = os.getcwd()
    main_dir = os.path.dirname(os.path.dirname(cwd))
    generate_participants(main_dir)
