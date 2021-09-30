# Author: https://github.com/AI-Cloud-and-Edge-Implementations/Project15-G4

from segment_files import SegmentFiles


def create_file_segments(metadata, file_range=30):
    segment_files = SegmentFiles(False, file_range=file_range)
    segment_files.process_segments(
        segment_files.ready_file_segments(metadata)
    )