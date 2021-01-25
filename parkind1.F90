! --------------------------------------------------------------------------------------------------
!> @brief Module for storing common data types.
!
!> @author
!> Sam Hatfield, ECMWF (samuel.hatfield@ecmwf.int)
!> Matthew Chantry, University of Oxford (matthew.chantry@physics.ox.ac.uk)
! --------------------------------------------------------------------------------------------------

MODULE PARKIND1
    IMPLICIT NONE
    
    ! Define IEEE 754 floating-point types
    INTEGER, PARAMETER :: SING = SELECTED_REAL_KIND(6, 37)
    INTEGER, PARAMETER :: DOUB = SELECTED_REAL_KIND(13, 300)
  
    ! Neural network precision
    INTEGER, PARAMETER :: NNP = DOUB
END MODULE

