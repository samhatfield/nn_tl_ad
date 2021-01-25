! --------------------------------------------------------------------------------------------------
!> @brief Tangent-linear and adjoint test for a neural network.
!
! Program for testing the tangent-linear and adjoint models of a neural network.
! Cited by the publication "Neural networks as the building blocks for tangent-linear and adjoint
! models in variational data assimilation", Hatfield, Dueben, Lopez, Geer, Chantry and Palmer (2021)
!
! Note that many arrays in this program have a superfluous singleton dimension (e.g. X1(NINP,1)).
! This is because the neural network code is designed to run over a batch of atmospheric columns.
! In this case though the batch size is always 1.
!
!> @author
!> Sam Hatfield, ECMWF (samuel.hatfield@ecmwf.int)
! --------------------------------------------------------------------------------------------------

PROGRAM MAIN
    USE NOGWDNN_MOD, ONLY: NN, NN_TL, NN_AD
    USE YONOGWDNN
    USE PARKIND1, ONLY: NNP, DOUB

    IMPLICIT NONE
    
    REAL(NNP) :: X1(NINP,1), X2(NINP,1), Y1(NOUT,1), Y2(NOUT,1), Y3(NOUT,1), DX(NINP,1), DY(NOUT,1)
    REAL(NNP) :: DX1(NINP,1), DX2(NINP,1), DY1(NOUT,1), DY2(NOUT,1)
    REAL(DOUB) :: LHS, RHS, DIFF
    INTEGER :: I

    ALLOCATE(INPUT_B(NWIDTH,1))
    ALLOCATE(OUTPUT_B(NOUT,1))
    ALLOCATE(INPUT(NWIDTH, NINP))
    ALLOCATE(OUTPUT(NOUT, NWIDTH))
    ALLOCATE(HIDDEN_B(NWIDTH, NHIDDEN,1))
    ALLOCATE(HIDDEN(NWIDTH, NWIDTH, NHIDDEN))
    
    ! Initialize all weight and bias arrays to random numbers
    CALL INITIALIZE_2D(INPUT)
    CALL INITIALIZE_3D(HIDDEN)
    CALL INITIALIZE_2D(OUTPUT)
    CALL INITIALIZE_2D(INPUT_B)
    CALL INITIALIZE_3D(HIDDEN_B)
    CALL INITIALIZE_2D(OUTPUT_B)

    ! ----------------------------------------------------------------------------------------------
    ! Tangent-linear test
    ! ----------------------------------------------------------------------------------------------
    ! The tangent-linear test evaluates the expression [NL(x+dx) - NL(x)] - TL(x)dx, where NL and TL
    ! are the nonlinear and tangent-linear models, respectively, x is a point about which the
    ! tangent-linear model is constructed and dx is a perturbation. This test is performed for
    ! smaller and smaller magnitudes of dx. If the result of the subtraction gets smaller and
    ! smaller then this indicates that the tangent-linear model is coded correctly with respect to
    ! the nonlinear model.
    ! Note that the tangent-linear model TL also calculates NL(x) and the code here returns this as
    ! an output. This is also compared here with NL(x) computed by the nonlinear model.
    ! ----------------------------------------------------------------------------------------------

    WRITE (*,'(A)') "Tangent-linear test"

    ! Fill linearisation point and perturbation vectors with random numbers
    CALL INITIALIZE_2D(X1)
    CALL INITIALIZE_2D(DX)
    
    DO I = 0, 10
        ! Construct second point
        X2 = X1 + DX

        WRITE (*,'(A,ES9.1)') "Size of perturbation ~= ", 10**(-REAL(I,NNP))
        
        ! Compute NL(x)
        CALL NN(1, X1, Y1)

        ! Compute NL(x+dx)
        CALL NN(1, X2, Y2)

        ! Compute TL(x)dx
        CALL NN_TL(1, X1, DX, Y3, DY)
        
        ! Compute [NL(x+dx) - NL(x)] - TL(x)dx
        DIFF = ABS(Y2(1,1) - Y1(1,1) - DY(1,1))

        WRITE (*,'(A,I3)') "Difference of 1st element around digit number ", NINT(ABS(LOG10(DIFF)))
        WRITE (*,'(A)') ''
        WRITE (*,'(A)') "Comparison of NL(x) from TL and NL models (should always be identical):"
        WRITE (*,'(5F21.17)') Y1(:,1)
        WRITE (*,'(5F21.17)') Y3(:,1)
        WRITE (*,'(A)')

        ! Reduce size of perturbation by a factor of 10
        DX = DX*0.1_NNP
    END DO

    WRITE (*,'(A)') "-----------------------------------------------------"
    WRITE (*,'(A)') ''

    ! ----------------------------------------------------------------------------------------------
    ! Adjoint test
    ! ----------------------------------------------------------------------------------------------
    ! The adjoint test evaluates the expressions <TL(x)dx, dy> and <dx, AD(x)dy>, where TL and AD
    ! are the tangent-linear and adjoint models, respectively, x is a point about which the TL and
    ! AD are constructed and dx and dy are arbitrary perturbations. The left-hand- and right-hand-
    ! sides should have about 16 consecutive digits in common when using double-precision. This test
    ! indicates that the adjoint model is coded correctly with respect to the tangent-linear model.
    ! ----------------------------------------------------------------------------------------------

    WRITE (*,'(A)') "Adjoint test"

    ! Initialize dx and dy
    CALL INITIALIZE_2D(DX1)
    CALL INITIALIZE_2D(DY2)
    DX1 = DX1*10.0_NNP**(-REAL(1, NNP))
    DY2 = DY2*10.0_NNP**(-REAL(1, NNP))

    ! Compute TL(x)dx
    CALL NN_TL(1, X1, DX1, Y1, DY1)

    ! Compute AD(x)dy
    CALL NN_AD(1, X1, DY2, DX2)

    WRITE (*,'(A)') "The following two numbers must be as similar as possible"

    ! Compute <TL(x)dx, dy>
    LHS = DOT_PRODUCT(DY1(:,1), DY2(:,1))

    ! Compute <dx, AD(x)dy>
    RHS = DOT_PRODUCT(DX1(:,1), DX2(:,1))

    WRITE (*,'(A,F21.17)') 'LHS', LHS
    WRITE (*,'(A,F21.17)') 'RHS', RHS
    DIFF = ABS(LHS - RHS)
    WRITE (*,'(A,I3)') "Difference of 1st element around digit number ", NINT(ABS(LOG10(DIFF)))
    
CONTAINS

! --------------------------------------------------------------------------------------------------

!> @brief Initialize a 2D array with Gaussian random numbers. The mean is 0 and the variance is 1.
!> @param[out] X the input array

SUBROUTINE INITIALIZE_2D(X)
    REAL(NNP), INTENT(OUT) :: X(:,:)
    
    INTEGER :: I, J
    
    DO I = 1, SIZE(X, 1)
        DO J = 1, SIZE(X, 2)
            X(I, J) = RANDN(0.0_NNP, 1.0_NNP)
        END DO
    END DO
END SUBROUTINE INITIALIZE_2D

! --------------------------------------------------------------------------------------------------

!> @brief Initialize a 3D array with Gaussian random numbers. The mean is 0 and the variance is 1.
!> @param[out] X the input array

SUBROUTINE INITIALIZE_3D(X)
    REAL(NNP), INTENT(OUT) :: X(:,:,:)
    
    INTEGER :: I, J, K
    
    DO I = 1, SIZE(X, 1)
        DO J = 1, SIZE(X, 2)
            DO K = 1, SIZE(X, 3)
                X(I, J, K) = RANDN(0.0_NNP, 1.0_NNP)
            END DO
        END DO
    END DO
END SUBROUTINE INITIALIZE_3D

! --------------------------------------------------------------------------------------------------

!> @brief Generated a random number from the Gaussin distribution, with given mean and standard
!> deviation.
!> @param[in] MEAN the mean of the distribution
!> @param[in] STDDEV the standard deviation of the distribution
!> @return RANDN the random number

FUNCTION RANDN(MEAN, STDEV)
    REAL(NNP), INTENT(IN) :: MEAN, STDEV
    REAL(NNP) :: U, V, RANDN
    REAL(NNP) :: RAND(2)

    CALL RANDOM_NUMBER(RAND)

    ! Box-Muller method
    U = (-2.0_NNP * LOG(RAND(1))) ** 0.5_NNP
    V =   2.0_NNP * 6.28318530718_NNP * RAND(2)
    RANDN = MEAN + STDEV * U * SIN(V)
END FUNCTION RANDN

! --------------------------------------------------------------------------------------------------

END PROGRAM MAIN

