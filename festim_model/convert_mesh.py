import meshio

def convert_med_to_xdmf(
    med_file,
    cell_file="3D_cad/8mm_2mm_nickel/mesh_domains.xdmf",
    facet_file="3D_cad/8mm_2mm_nickel/mesh_boundaries.xdmf",
    cell_type="triangle",
    facet_type="line",
):
    """Converts a MED mesh to XDMF
    Args:
        med_file (str): the name of the MED file
        cell_file (str, optional): the name of the file containing the
            volume markers. Defaults to "mesh_domains.xdmf".
        facet_file (str, optional): the name of the file containing the
            surface markers.. Defaults to "mesh_boundaries.xdmf".
        cell_type (str, optional): The topology of the cells. Defaults to "tetra".
        facet_type (str, optional): The topology of the facets. Defaults to "triangle".
    Returns:
        dict, dict: the correspondance dict, the cell types
    """
    msh = meshio.read(med_file)

    correspondance_dict = msh.cell_tags

    cell_data_types = msh.cell_data_dict["cell_tags"].keys()

    for mesh_block in msh.cells:
        if mesh_block.type == cell_type:

            meshio.write_points_cells(
                cell_file,
                msh.points,
                [mesh_block],
                cell_data={"f": [-1 * msh.cell_data_dict["cell_tags"][cell_type]]},
            )
        elif mesh_block.type == facet_type:
            meshio.write_points_cells(
                facet_file,
                msh.points,
                [mesh_block],
                cell_data={"f": [-1 * msh.cell_data_dict["cell_tags"][facet_type]]},
            )

    return correspondance_dict, cell_data_types


if __name__ == "__main__":


    filename = "3D_cad/8mm_2mm_nickel/8mm_2mm_nickel.med"
    correspondance_dict, cell_data_types = convert_med_to_xdmf(
        filename)
    print(correspondance_dict)