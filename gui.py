import streamlit as st

from source.data.data import Dataset
from source import DATASETS_LIST, DATASETS_MAP
from source.gui.regressor_vm import RegressorViewModel
from source.gui.classifier_vm import ClassifierViewModel
from source.setup.configuration import ConfigurationSetup


# tab title
st.set_page_config(page_title='DeepLearningFramework', layout='wide')

# page heading
st.write('# Automated Deep Learning')

#
with st.form('deep_learner'):
    # row 1 set-up
    _, col2, col3 = st.columns([1,3,1])

    # select 
    model_selection = col2.selectbox(
        label='Select model',
        options=DATASETS_LIST,
        index=0
    )

    submit_button = st.form_submit_button('Run')

# upon form submission
if submit_button:
    # configuration
    config = ConfigurationSetup(manual=DATASETS_MAP[model_selection])
    dataset = Dataset(configuration=config)

    # model run selected model
    match model_selection:
        case 'Regressor':
            RegressorViewModel(data=dataset)
        case 'Classifier':
            ClassifierViewModel(data=dataset)
        case _:
            pass
