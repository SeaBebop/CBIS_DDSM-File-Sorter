create table metadata_roi_only (
select * from edited_metadata
inner join  calc_case_test on edited_metadata.Subject_ID = calc_case_test.ROI_mask_file_path
where Series_Description = 'ROI mask images'
Union ALL
select * from edited_metadata
inner join  calc_case_training on edited_metadata.Subject_ID = calc_case_training.ROI_mask_file_path
where Series_Description = 'ROI mask images'
Union ALL
select * from edited_metadata
inner join mass_case_test on edited_metadata.Subject_ID = mass_case_test.ROI_mask_file_path
where Series_Description = 'ROI mask images'
Union All
select * from edited_metadata
inner join mass_case_training on edited_metadata.Subject_ID = mass_case_training.ROI_mask_file_path
where Series_Description = 'ROI mask images'
)