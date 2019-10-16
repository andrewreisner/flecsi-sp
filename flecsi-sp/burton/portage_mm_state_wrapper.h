/*~--------------------------------------------------------------------------~*
 * Copyright (c) 2016 Los Alamos National Laboratory, LLC
 * All rights reserved
 *~--------------------------------------------------------------------------~*/
////////////////////////////////////////////////////////////////////////////////
/// \file
////////////////////////////////////////////////////////////////////////////////

#pragma once

// user includes

// library includes
#include <portage/support/portage.h>
//#include <ristra/math/array.h>

// system includes
#include <utility>
#include <cstring>
#include <string>
#include <map>
#include <cstdarg>

namespace flecsi_sp {
namespace burton {

////////////////////////////////////////////////////////////////////////////////
/// \brief Provides access to data stored in Flecsi_State
////////////////////////////////////////////////////////////////////////////////
template< typename M >
class portage_mm_state_wrapper_t {

  //============================================================================
  // Typedefs
  //============================================================================

  //! \brief The mesh type
  using mesh_t = M;
  //! \brief the size type
  using size_t = typename mesh_t::size_t;
  //! \brief the real type
  using real_t = typename mesh_t::real_t;
  
  using vector_t = typename mesh_t::vector_t;


  //! \brief The entity kind type
  using entity_kind_t = Portage::Entity_kind;
  //! \brief The entity type 
  using entity_type_t = Portage::Entity_type;
  //! \brief The field type
  using field_type_t = Portage::Field_type;

  using byte_t = unsigned char;


  //============================================================================
  // Member Variables
  //============================================================================
  
  //! \brief the flecsi mesh pointer
  mesh_t * mesh_ = nullptr;

  struct data_t {
    entity_kind_t entity_kind = entity_kind_t::UNKNOWN_KIND;
    field_type_t field_type = field_type_t::UNKNOWN_TYPE_FIELD;
    size_t data_size = 0;
    std::vector<byte_t> data;
  };

  std::map < std::string, data_t > var_map_;

  std::vector<std::vector<int>> mat_cells_;
  std::vector<std::vector<int>> cell_mat_ids_;
  std::vector<std::vector<int>> cell_mat_offsets_;

  std::vector<int> mat_data_offsets_;

  int number_materials_=0;

public:

  //============================================================================
  // Constructors
  //============================================================================

  //!  \brief Default constructor.
  //!  \param[in] mesh The minimum coordinates of the domain.
  //!  \param[in] mesh The minimum coordinates of the domain.
  explicit portage_mm_state_wrapper_t(mesh_t & mesh) : mesh_(&mesh)
    {}

  //! Default constructor deleted
  portage_mm_state_wrapper_t() = default;

  //! Default copy constructor
  portage_mm_state_wrapper_t(const portage_mm_state_wrapper_t &) = default;

  //! Default assignment operator
  portage_mm_state_wrapper_t & operator=(const portage_mm_state_wrapper_t &) = default;

  //============================================================================
  // Public Members
  //============================================================================

  //! \brief Add a variable of entity type (cell) 
  //! field that needs to be remapped to the variable map
  template< typename T >
  auto add_cell_field(
    std::string var_name,
    entity_kind_t entity=entity_kind_t::CELL,
    field_type_t type=field_type_t::MESH_FIELD)
  {
    auto ret = var_map_.emplace( var_name, data_t{} );
    if (!ret.second) 
      THROW_RUNTIME_ERROR( var_name << " already registered" );
    auto & entry = ret.first->second;
    entry.entity_kind = entity;
    entry.field_type = type;
    entry.data_size = sizeof(T);

    entry.data.resize(mat_data_offsets_.back() * sizeof(T));
    return reinterpret_cast<T*>(entry.data.data());
  }

  //! \brief Number of materials in problem
  int num_materials() const {
    return number_materials_;
  }

  //! \brief Set the number of materials in problem
  void set_materials(
    int num_mats,
    const std::vector< std::vector<int> > & mat_cells,
    const std::vector< std::vector<int> > & cell_mats,
    const std::vector< std::vector<int> > & cell_mat_offsets)
  {
    number_materials_ = num_mats;
    mat_cells_ = mat_cells;
    cell_mat_ids_ = cell_mats;
    cell_mat_offsets_ = cell_mat_offsets;

    mat_data_offsets_.resize(num_mats+1);
    mat_data_offsets_[0] = 0;
    for ( int i=0; i<num_mats; ++i )
      mat_data_offsets_[i+1] = mat_data_offsets_[i] + mat_cells[i].size();

  }

  //! \brief Name of material
  std::string material_name(int matid) const {
    // return something else if you wanted to keep track of whether or not
    // the material has been added
    assert(matid >= 0 && matid < num_materials());
    return std::to_string(matid);
  }

  //! \brief Get number of cells containing a particular material
  //! \param matid    Index of material (0, num_materials()-1)
  //! \return         Number of cells containing material 'matid'
  int mat_get_num_cells(int matid) const {
    assert(matid >= 0 && matid < num_materials());
    return mat_cells_[matid].size();
  }

  //! \brief Get cell indices containing a particular material
  //! \param matid    Index of material (0, num_materials()-1)
  //! \param matcells Cells containing material 'matid'
  void mat_get_cells(int matid, std::vector<int> *matcells) const{
    assert(matid >= 0 && matid < num_materials());
    *matcells = mat_cells_[matid];
  }

  //! \brief Get number of materials contained in a cell
  //! \param cellid  Index of cell in mesh
  //! \return        Number of materials in cell
  int cell_get_num_mats(int cellid) const {
    return cell_mat_ids_[cellid].size();
  }

  //! \brief Get the IDs of materials in a cell
  //! \param cellid    Index of cell in mesh
  //! \param cellmats  Indices of materials in cell
  void cell_get_mats(int cellid, std::vector<int> *cellmats) const {
    *cellmats = cell_mat_ids_[cellid];
  }

  //! \brief Get the local index of mesh cell in material cell list
  //! \param meshcell    Mesh cell ID
  //! \param matid       Material ID
  //! \return             Local cell index in material cell list
  int cell_index_in_material(int meshcell, int matid) const {
    const auto & mat_ids = cell_mat_ids_[meshcell];
    auto it = std::find( mat_ids.begin(), mat_ids.end(), matid );
    if (it == mat_ids.end())
      THROW_RUNTIME_ERROR( "MAT_ITER not found " << meshcell << " " << matid );

    auto offset = std::distance(mat_ids.begin(), it);
    auto index = cell_mat_offsets_[meshcell][offset];
    return index;
  }

  
  //! \brief Type of field (MESH_FIELD or MULTIMATERIAL_FIELD)
  //! \param[in] onwhat   Entity_kind that field is defined on
  //! \param[in] varname  Name of field
  //! \return             Field_type
  field_type_t field_type(entity_kind_t on_what, std::string const& var_name) const
  {
  
    auto it = var_map_.find(var_name);
    if ( it == var_map_.end() )
      THROW_RUNTIME_ERROR( " Could not find state variable field type " <<
        var_name << " on " << on_what );
    return it->second.field_type;
  }
  
  //! \brief Get the entity type on which the given field is defined
  //! \param[in] var_name The string name of the data field
  //! \return The Entity_kind enum for the entity type on which the field is defined
  //!
  //! \todo  THIS ASSUMES ONLY DOUBLE VECTORS - WE HAVE TO ACCOUNT FOR OTHER TYPES
  //!        OR WE HAVE TO GENERALIZE THE FIND FUNCTION!!!
  //! \todo  THIS ALSO DOES NOT CHECK FOR OTHER ENTITY TYPES LIKE EDGE, FACE,
  //!        SIDE, WEDGE AND CORNER
  entity_kind_t get_entity(std::string const& var_name) const 
  {
    auto it = var_map_.find(var_name);
    if ( it == var_map_.end())
      return entity_kind_t::UNKNOWN_KIND;
    else
      return it->second.entity_kind;
  }

  //! \brief Get pointer to scalar data
  //! \param[in] on_what The entity type on which to get the data
  //! \param[in] var_name The string name of the data field
  //! \param[in,out] data A pointer to an array of data
  template <class T>
  void mesh_get_data(entity_kind_t on_what, std::string const& var_name,
    T ** data)
  {
    auto it = var_map_.find(var_name);
    if ( it == var_map_.end() )
      THROW_RUNTIME_ERROR( " Could not find state variable data for " <<
          var_name );
    *data = reinterpret_cast<T*>(it->second.data.data());
  }
  
  //! \brief Get pointer to scalar data
  //! \param[in] on_what The entity type on which to get the data
  //! \param[in] var_name The string name of the data field
  //! \param[in,out] data A pointer to an array of data
  template <class T>
  void mesh_get_data(entity_kind_t on_what, std::string const& var_name,
    T const ** data) const
  {
    auto it = var_map_.find(var_name);
    if ( it == var_map_.end() )
      THROW_RUNTIME_ERROR( " Could not find state variable data for " <<
          var_name );
    *data = reinterpret_cast<const T*>(it->second.data.data());
  }
  
  //! \brief Get pointer to read-only scalar data for a particular material
  //! \param[in] on_what The entity type on which to get the data
  //! \param[in] var_name The string name of the data field
  //! \param[in] matid   Index (not unique identifier) of the material
  //! \param[out] data   vector containing the values corresponding to cells in the material
  template< typename T >
  void mat_get_celldata(std::string const& var_name, int matid, T * * const values) const
  {
    
    // add it if its not in the map
    auto it = var_map_.find(var_name);
    if (it==var_map_.end())
      THROW_RUNTIME_ERROR( " Could not find state variable data for " <<
          var_name );

    auto offset = mat_data_offsets_[matid];
    *values = reinterpret_cast<T*>(it->second.data.data()) + offset;
  }


  //! \brief Get pointer to read-write scalar data for a particular material
  //! \param[in] on_what The entity type on which to get the data
  //! \param[in] var_name The string name of the data field
  //! \param[in] matid   Index (not unique identifier) of the material
  //! \param[out] data   vector containing the values corresponding to cells in the material
  template< typename T >
  void mat_get_celldata(std::string const& var_name, int matid, T **values)
  {
    
    // add it if its not in the map
    auto it = var_map_.find(var_name);
    if ( it != var_map_.end() ){
      auto ret = var_map_.emplace( var_name, data_t{} );
      it = ret.first;
      it->second.data_size = sizeof(T);
    }

    auto offset = mat_data_offsets_[matid];
    auto size = mat_cells_[matid].size();
    auto required_size = offset + size;

    // get a reference to the data entry
    auto & entry = it->second;
    auto data_size = entry.data_size;
    auto & data = entry.data;
    auto current_size = data.size() / data_size;

    // check the size to see if it has been sized for this material already.
    // If it hasnt, then resize it
    if (required_size > current_size) data.resize(required_size * data_size);

    // set pointer
    *values = reinterpret_cast<T*>(data.data()) + offset;
  }

  //! \brief Add a scalar multi-valued data field on cells and add
  //! data to one of its materials
  //! \param[in] var_name The name of the data field
  //! \param[in] matid  Index of material in the problem
  //! \param[in] layout Data layout - 
  //! \param[in] values Initialize with this array of values
  //!
  //! Subsequent calls to this function with the same name will find the added
  //! field and just add the data.

  template< typename T >
  void mat_add_celldata(std::string const& var_name, int matid,
      T const * values)
  {

    T * data_ptr;
    mat_get_celldata( var_name, matid, &data_ptr );

    // if its our pointer, then do nothing.  if its not, then copy.
    if (data_ptr != values) {

      auto size = mat_cells_[matid].size();
      for (size_t i=0; i < size; ++i) data_ptr[i] = values[i];

    }

  }

  //! \brief Add cells to material (or add material to cells)
  //! \param[in] matid  Material ID
  //! \param[in] newcells Vector of new cells in material
  void mat_add_cells(int matid, std::vector<int> const & newcells)
  {

    // store the last material and increment materials
    number_materials_ = matid + 1;

    // this version only sets the material cells?

    // setting material cells is easy
    mat_cells_.resize( number_materials_ );
    mat_cells_[matid] = newcells;

    // need largest cell id for resizing
    auto it = std::max_element( newcells.begin(), newcells.end() );
    auto nc = *it + 1;


    // fill in all material ids, assume that the material id is monotonically
    // increasing
    cell_mat_ids_.resize( nc );
    for ( auto c : newcells )
      cell_mat_ids_[c].push_back(matid);

    // fill in all material mesh ids, assume that the material id and cell id
    // are increasing monotonically
    cell_mat_offsets_.resize( nc );
    size_t mat_cell_id{0};
    for ( auto c : newcells ) {
      cell_mat_offsets_[c].push_back(mat_cell_id);
      mat_cell_id++;
    }
 
    // set offsets for next rang of materials.
    // Just redo them all since the number of materials is small
    mat_data_offsets_.resize(number_materials_+1);

    mat_data_offsets_[0] = 0;
    for ( auto i=0; i<number_materials_; ++i )
      mat_data_offsets_[i+1] = mat_data_offsets_[i] + mat_cells_[i].size();
  }


  //! \brief Add a material to state
  //! \param[in] matname  Name of material
  //! \param[in] matcells Cells containing the material
  void add_material(std::string const& matname, std::vector<int> const& matcells)
  {
    // supposed to create field storage here, but we assume it has already been
    // sized for now.
    auto matid = std::stoi( matname );
    mat_add_cells( matid, matcells );
  }

  //! Get the size of a particular quantity
  //! WARNING: this may not be the correct value for multi-material quantities
  int get_data_size(entity_kind_t on_what, std::string const& var_name) const
  {
    return mesh_->num_cells();
  }

  //! @brief Get the data type of the given field
  //! @param[in] var_name The string name of the data field
  //! @return A reference to the type_info struct for the field's data type
  const std::type_info& get_data_type(std::string const& var_name) const {
    return typeid(double);  // thats the only type we can represent
  }
  
 };  // Flecsi_State_Wrapper
 
} // namespace burton
} // namespace flecsi_sp
