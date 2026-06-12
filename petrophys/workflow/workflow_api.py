from enum import Enum
from petrophys.workflow.dynamic_workflow import WorkflowStep

STEP_META: dict[str, dict] = {
    WorkflowStep.TEMPERATURE: {
        "description": "Compute formation temperature vs. depth using surface temp + gradient.",
        "inputs":  ["DEPTH"],
        "outputs": ["TEMP"],
        "required": True,
    },
    WorkflowStep.BAD_HOLE: {
        "description": "Flag washed-out / bad-hole intervals using caliper vs. bit-size threshold.",
        "inputs":  ["CALI", "BS", "DRHO", "RHOB", "PEF"],
        "outputs": ["BAD_HOLE"],
        "required": True,
    },
   WorkflowStep.VCLAY: {
        "description": "Estimate clay volume from GR, SP, or neutron-density crossplot using the selected method.",
        "inputs":  ["GR", "SP", "NPHI", "RHOB"],
        "outputs": ["VCL", "VCLGR", "VCLSP", "VCLND"],
        "required": True,
        "methods": ["larionov_tertiary", "larionov_older", "clavier", "linear", "steiber"],
        "api": {
            "vcl_from_gr":  {"signature": "vcl_from_gr(gr, gr_min, gr_max, method)", "note": "GR-based, 5 methods"},
            "vcl_from_sp":  {"signature": "vcl_from_sp(sp, sp_max, sp_min)",          "note": "SP-based"},
            "vcl_from_nd":  {"signature": "vcl_from_nd(nphi, rhob, ...)",              "note": "Neutron-density crossplot"},
        },
    },
    WorkflowStep.RCL: {
        "description": "Resolve clay-bound resistivity correction.",
        "inputs":  ["RT", "VCLAY"],
        "outputs": ["RCL"],
        "required": False,
    },
    WorkflowStep.LITHOLOGY: {
        "description": "Identify lithology / matrix type using multi-mineral solver.",
        "inputs":  ["RHOB", "NPHI"],
        "outputs": ["LITHOLOGY", "MATRIX"],
        "required": True,
        "methods": ["limestone", "sandstone", "dolomite"],
    },
    WorkflowStep.RMF: {
        "description": "Estimate mud filtrate resistivity from mud weight and temperature.",
        "inputs":  ["TEMP", "MUD_WEIGHT"],
        "outputs": ["RMF"],
        "required": False,
    },
    WorkflowStep.RW: {
        "description": "Compute formation water resistivity (Arps / catalogue).",
        "inputs":  ["TEMP"],
        "outputs": ["RW"],
        "required": True,
    },
    WorkflowStep.INVASION: {
        "description": "Detect mud-filtrate invasion using shallow vs. deep resistivity ratio.",
        "inputs":  ["MSFL", "RT"],
        "outputs": ["INVASION_FLAG"],
        "required": False,
    },
    WorkflowStep.HC_CORRECTION: {
        "description": "Apply hydrocarbon correction to neutron-density porosity.",
        "inputs":  ["NPHI", "RHOB", "SW_UNCORR"],
        "outputs": ["NPHI_HC", "RHOB_HC"],
        "required": False,
    },
    WorkflowStep.POROSITY: {
        "description": "Compute effective porosity (PHIE) from density, neutron, or sonic.",
        "inputs":  ["RHOB", "NPHI"],
        "outputs": ["PHIT", "PHIE"],
        "required": True,
        "methods": ["density", "neutron", "sonic", "density_neutron"],
    },
    WorkflowStep.POROSITY_REFINEMENT: {
        "description": "Recompute porosity after inferred lithology is available, using lithology-specific matrix endpoints.",
        "inputs":  ["RHOB", "NPHI", "DTC", "LITHO"],
        "outputs": ["PHIN", "PHID", "PHIS", "PHIT", "PHIE"],
        "required": False,
        "methods": ["density", "neutron", "sonic", "density_neutron"],
    },
    WorkflowStep.SATURATION: {
        "description": "Compute water saturation (Sw) via Archie or Simandoux.",
        "inputs":  ["RT", "PHIE", "RW", "VCL"],
        "outputs": ["SW", "SHC"],
        "required": True,
        "methods": ["archie", "simandoux", "indonesia", "waxman_smits"],
    },
    WorkflowStep.PERMEABILITY: {
        "description": "Estimate permeability from porosity and irreducible water saturation.",
        "inputs":  ["PHIE", "SW"],
        "outputs": ["PERM"],
        "required": True,
        "methods": ["coates", "timur", "kozeny"],
    },
    WorkflowStep.BVW: {
        "description": "Compute bulk volume water (BVW = PHIE × Sw).",
        "inputs":  ["PHIE", "SW"],
        "outputs": ["BVW"],
        "required": False,
    },
    WorkflowStep.MHI: {
        "description": "Moving Hydrocarbon Index — ratio of BVW to porosity trend.",
        "inputs":  ["BVW", "PHIE"],
        "outputs": ["MHI"],
        "required": False,
    },
    WorkflowStep.NET_PAY: {
        "description": "Flag net-pay intervals using VCL, PHIE, and Sw cutoffs.",
        "inputs":  ["VCL", "PHIE", "SW"],
        "outputs": ["NET_PAY_FLAG", "NET_PAY_THICKNESS"],
        "required": True,
    },
}


CATEGORY_STEP_MAP: dict[str, list[WorkflowStep]] = {
    "all_categories": [
        WorkflowStep.TEMPERATURE, WorkflowStep.BAD_HOLE, WorkflowStep.VCLAY,
        WorkflowStep.RCL, WorkflowStep.LITHOLOGY, WorkflowStep.RMF, WorkflowStep.RW,
        WorkflowStep.INVASION, WorkflowStep.HC_CORRECTION, WorkflowStep.POROSITY,
        WorkflowStep.POROSITY_REFINEMENT,
        WorkflowStep.SATURATION, WorkflowStep.PERMEABILITY, WorkflowStep.BVW,
        WorkflowStep.MHI, WorkflowStep.NET_PAY,
    ],
    "porosity": [
        WorkflowStep.TEMPERATURE, WorkflowStep.BAD_HOLE, WorkflowStep.POROSITY,
        WorkflowStep.POROSITY_REFINEMENT,
    ],
    "saturation": [
        WorkflowStep.TEMPERATURE, WorkflowStep.BAD_HOLE,
        WorkflowStep.RW, WorkflowStep.SATURATION,
    ],
    "permeability": [
        WorkflowStep.TEMPERATURE, WorkflowStep.BAD_HOLE, WorkflowStep.PERMEABILITY,
    ],
    "lithology": [
        WorkflowStep.TEMPERATURE, WorkflowStep.BAD_HOLE, WorkflowStep.VCLAY,
        WorkflowStep.POROSITY, WorkflowStep.LITHOLOGY, WorkflowStep.POROSITY_REFINEMENT,
    ],
    "hydrocarbon": [
        WorkflowStep.TEMPERATURE, WorkflowStep.BAD_HOLE, WorkflowStep.SATURATION, WorkflowStep.BVW, WorkflowStep.MHI, WorkflowStep.NET_PAY,
    ],
}


