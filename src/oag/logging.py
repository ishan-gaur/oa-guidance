import collections
import pickle as pkl
from warnings import warn
from oag.constants import DB_FOLDER

def log_file_path(experiment_name, **kwargs):
    data_file_stem = experiment_name
    def add_arg(s, func=(lambda s, v: f"{s}_{v}"), just_val=False, skip_if=(lambda v: False)):
        val = kwargs.get(s, None)
        if skip_if(val):
            return
        elif val is None:
            raise ValueError(f"{s} must be provided to get log-file path.")

        nonlocal data_file_stem
        if just_val:
            func = lambda s, v: str(v)
        data_file_stem += f"_{func(s, val)}"

    # add_arg("level_str", just_val=True)
    add_arg("mask_level")
    add_arg("guidance_temp")
    add_arg("x1_temp")
    add_arg("use_tag", func=lambda s, v: "tag" if v else "exact")
    add_arg("top_ecs", just_val=True, skip_if=lambda v: v is None or v)
    add_arg("min_ecs", just_val=True, skip_if=lambda v: v is None or v)
    add_arg("debug", func=lambda s, v: "debug", skip_if=lambda v: v is None or not v)
    data_file = DB_FOLDER / f"{data_file_stem}.pkl"
    return data_file

class Logger:
    def __init__(self, fields, verbose=False): # fields can be None, but I want that to be explicit
        self.verbose = verbose
        self.fields = set(fields) if fields else set()
        self.data = collections.defaultdict( # for each sample
            lambda: collections.defaultdict( # for each field
                list
            )
        )

    def log(self, field, value, sample_idx=0):
        if self.fields and field not in self.fields:
            if self.verbose:
                warn(f"Field {field} not in fields. Ignoring.")
            return
        self.data[sample_idx][field].append(value)

    def get_sample(self, sample_idx):
        return self.data[sample_idx]

    def get_all(self):
        return self.data

    def add_field(self, field):
        self.fields.add(field)

    def to_file(self, filename):
        for sample_idx, sample in self.data.items():
            self.data[sample_idx] = dict(sample) # Convert defaultdict to dict for serialization
        # Convert the entire data structure to a regular dict for serialization
        self.data = dict(self.data)
        with open(filename, "wb") as f:
            pkl.dump(self.data, f)
        self.data = Logger.from_file(filename).data # Reload the data to ensure it's in the correct format


    def assert_samples_logged(self, samples):
        if set(samples) != set(self.data.keys()):
            raise ValueError(f"Samples logged {set(self.data.keys())} do not match samples {set(samples)}")
    
    def assert_fields_logged(self):
        if not self.fields: # If no fields are specified to be logged, this check is trivially true.
            return

        missing_fields_summary = collections.defaultdict(int)
        num_samples_logged = len(self.data)

        # Iterate through each field that is expected to be logged
        for field_name in self.fields:
            samples_missing_this_field = 0
            # For the current expected field, check all logged samples
            # If self.data is empty, this inner loop will not run,
            # samples_missing_this_field will remain 0, and this field won't be marked as missing.
            for sample_idx in self.data: 
                if field_name not in self.data[sample_idx]:
                    samples_missing_this_field += 1
            
            if samples_missing_this_field > 0:
                missing_fields_summary[field_name] = samples_missing_this_field

        if missing_fields_summary:
            # Convert defaultdict to dict for a cleaner error message
            error_details = dict(missing_fields_summary)
            message = (
                f"Fields missing from one or more samples. "
                f"Total samples logged: {num_samples_logged}.\n"
                f"Missing fields (field: number of samples missing it): {error_details}"
            )
            raise ValueError(message)

    @classmethod
    def from_file(cls, filename):
        with open(filename, "rb") as f:
            data = pkl.load(f)
        sample_0 = next(iter(data.values()))
        fields = sample_0.keys()
        def_dict_data = collections.defaultdict(
            lambda: collections.defaultdict(list)
        )
        # convert back to defaultdict with minimal memory overhead
        # by copying on sample at a time and then deleting the original
        while len(data) > 0:
            sample_idx, sample_dict = next(iter(data.items()))
            def_dict_data[sample_idx].update(sample_dict)
            del data[sample_idx]

        logger = cls(fields=fields)
        logger.data = def_dict_data
        return logger

    def __repr__(self):
        return f"Logger(fields={self.fields}, n_samples={len(self.data)}, verbose={self.verbose})"