# PP_2023 MCCA
The goal of the project was to explore the inter-subject analysis capabilities given by the M-CCA algorithm applied to an MEG data set. To achieve this, we selected an MEG data set that contained recordings of subjects listening to naturalistic speech stimuli, so we could track the sound envelope of this stimuli with the shared space created by the M-CCA procedure. Research has demonstrated that the sound envelope is closely related to the neural activity in the auditory cortex. Using the M-CCA procedure we were able to extract the more relevant components that were active in the data, and that were shared between participants. Correlation analysis shows that the relationship between the envelope and the canonical components is similar throughout participants. When fitting a linear model using the envelope as a dependent and the canonical components as independent it shows that the components have a different predictive power and explained variance. The sound envelope shows a relationship with our components, so the envelope could be used as a reference point of recording auditory stimuli. The M-CCA procedure shows relevance for inter-subject data analysis in MEG data.

#### The code was created by José Chonay as part of the Practical Project at the Applied Neurocognitive Psychology Lab at Oldenburg Universitat under the supervision of M.Sc. Leo Michalcke and Prof. Dr. Jochem Rieger.
#### The structure of the repository consists of:
[ ] Pre processing the MEG data and epoching. \
[ ] Extracting the sound envelope.\
[ ] Calculating the canonical components.\
[ ] Calculating the correlation coefficient between the components and the sound envelope.\
[ ] Testing the significance of the correlation coefficients across subjects.\
[ ] Reconstructing the sound envelope using the canonical components as a regressor.\
[ ] Visualization of the canonical components in the MEG sensor space.\

### The raw data used for this project can be found at: https://osf.io/ag3kj/