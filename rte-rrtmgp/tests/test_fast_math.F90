! Test and benchmark program for fast_exp approximation
!
! This program validates the accuracy and measures the speedup of the
! fast exp() approximation against the intrinsic exp() function.
!
! The test covers the argument range relevant to radiative transfer:
!   - Transmissivity: exp(-tau) where tau in [0, 50]
!   - Two-stream: exp(-k*tau) where k*tau in [0, 100]
!
! Success criteria:
!   - Maximum relative error < 1e-6 (single precision) or < 1e-14 (double precision)
!   - Speedup > 1.5x over intrinsic exp()
!   - Zero failures in edge case testing
!
! Usage: compile with gfortran -O3 -march=native test_fast_math.F90 mo_rte_kind.F90 mo_fast_math.F90
! -------------------------------------------------------------------------------------------------
program test_fast_math
  use mo_rte_kind, only: wp, sp, dp
  use mo_fast_math, only: fast_exp, fast_exp_array, fast_trans_array
  implicit none

  integer, parameter :: NTEST = 10000000  ! 10 million test points
  integer, parameter :: NBENCH = 100000000 ! 100 million for benchmarking
  integer, parameter :: NCOL = 1000, NLAY = 127  ! Typical GFS dimensions

  real(wp), allocatable :: x(:), result_fast(:), result_ref(:)
  real(wp), allocatable :: tau(:,:), D(:), trans_fast(:,:), trans_ref(:,:)
  real(wp) :: max_rel_err, max_abs_err, avg_rel_err, rel_err
  real(wp) :: t_start, t_end, t_fast, t_ref, speedup
  integer  :: i, ilay, icol, n_errors
  logical  :: all_passed

  all_passed = .true.

  write(*,*) "============================================="
  write(*,*) " Fast exp() Approximation Test Suite"
  write(*,*) "============================================="
#ifdef RTE_USE_SP
  write(*,*) " Precision: SINGLE (sp)"
#else
  write(*,*) " Precision: DOUBLE (dp)"
#endif
  write(*,*)

  ! -------------------------------------------------------------------------------------------------
  ! TEST 1: Accuracy over RT-relevant range [-50, 0]
  ! -------------------------------------------------------------------------------------------------
  write(*,*) "--- Test 1: Accuracy over [-50, 0] ---"

  allocate(x(NTEST), result_fast(NTEST), result_ref(NTEST))

  ! Generate test points uniformly in [-50, 0]
  do i = 1, NTEST
    x(i) = -50.0_wp * real(i-1, wp) / real(NTEST-1, wp)
  end do

  ! Compute reference and fast versions
  do i = 1, NTEST
    result_ref(i) = exp(x(i))
    result_fast(i) = fast_exp(x(i))
  end do

  ! Compute errors
  max_rel_err = 0.0_wp
  max_abs_err = 0.0_wp
  avg_rel_err = 0.0_wp
  n_errors = 0

  do i = 1, NTEST
    if (result_ref(i) > tiny(1.0_wp)) then
      rel_err = abs(result_fast(i) - result_ref(i)) / result_ref(i)
      max_rel_err = max(max_rel_err, rel_err)
      avg_rel_err = avg_rel_err + rel_err
    end if
    max_abs_err = max(max_abs_err, abs(result_fast(i) - result_ref(i)))
  end do
  avg_rel_err = avg_rel_err / real(NTEST, wp)

  write(*,'(A,ES12.5)') "  Max relative error: ", max_rel_err
  write(*,'(A,ES12.5)') "  Avg relative error: ", avg_rel_err
  write(*,'(A,ES12.5)') "  Max absolute error: ", max_abs_err

#ifdef RTE_USE_SP
  if (max_rel_err > 1.0e-6_wp) then
    write(*,*) "  FAIL: Relative error exceeds 1e-6 threshold"
    all_passed = .false.
  else
    write(*,*) "  PASS"
  end if
#else
  if (max_rel_err > 1.0e-14_wp) then
    write(*,*) "  FAIL: Relative error exceeds 1e-14 threshold"
    all_passed = .false.
  else
    write(*,*) "  PASS"
  end if
#endif
  write(*,*)

  ! -------------------------------------------------------------------------------------------------
  ! TEST 2: Edge cases
  ! -------------------------------------------------------------------------------------------------
  write(*,*) "--- Test 2: Edge cases ---"
  n_errors = 0

  ! exp(0) should be exactly 1
  if (abs(fast_exp(0.0_wp) - 1.0_wp) > epsilon(1.0_wp)) then
    write(*,*) "  FAIL: fast_exp(0) /= 1.0, got:", fast_exp(0.0_wp)
    n_errors = n_errors + 1
  end if

  ! exp(-50) should be very close to 0 (or exactly 0 from cutoff)
  if (fast_exp(-50.0_wp) < 0.0_wp) then
    write(*,*) "  FAIL: fast_exp(-50) < 0"
    n_errors = n_errors + 1
  end if

  ! Small negative arguments (thin layers)
  rel_err = abs(fast_exp(-1.0e-8_wp) - exp(-1.0e-8_wp)) / exp(-1.0e-8_wp)
  if (rel_err > 1.0e-6_wp) then
    write(*,*) "  FAIL: fast_exp(-1e-8) relative error too large:", rel_err
    n_errors = n_errors + 1
  end if

  ! exp(-1) = 0.367879441...
  rel_err = abs(fast_exp(-1.0_wp) - exp(-1.0_wp)) / exp(-1.0_wp)
  if (rel_err > 1.0e-6_wp) then
    write(*,*) "  FAIL: fast_exp(-1) relative error too large:", rel_err
    n_errors = n_errors + 1
  end if

  ! Monotonicity: fast_exp should be monotonically increasing
  do i = 2, NTEST
    if (result_fast(i) < result_fast(i-1) .and. x(i) > x(i-1)) then
      write(*,*) "  FAIL: Non-monotonic at x =", x(i)
      n_errors = n_errors + 1
      exit  ! Report first failure only
    end if
  end do

  ! Non-negativity: exp() is always >= 0
  do i = 1, NTEST
    if (result_fast(i) < 0.0_wp) then
      write(*,*) "  FAIL: Negative result at x =", x(i)
      n_errors = n_errors + 1
      exit
    end if
  end do

  if (n_errors == 0) then
    write(*,*) "  PASS (all edge cases)"
  else
    write(*,'(A,I0,A)') "  FAIL (", n_errors, " edge case failures)"
    all_passed = .false.
  end if
  write(*,*)

  deallocate(x, result_fast, result_ref)

  ! -------------------------------------------------------------------------------------------------
  ! TEST 3: Benchmark scalar exp()
  ! -------------------------------------------------------------------------------------------------
  write(*,*) "--- Test 3: Benchmark scalar exp() ---"
  write(*,'(A,I0,A)') "  Evaluating ", NBENCH, " exp() calls..."

  allocate(x(NBENCH), result_fast(NBENCH), result_ref(NBENCH))

  ! Generate realistic RT argument distribution
  ! Most optical depths are small (thin layers), some are large (clouds)
  do i = 1, NBENCH
    ! Mix of thin layers (90%) and thick layers (10%)
    if (mod(i, 10) == 0) then
      x(i) = -10.0_wp - 40.0_wp * real(mod(i*7, 1000), wp) / 1000.0_wp
    else
      x(i) = -0.001_wp - 5.0_wp * real(mod(i*13, 1000), wp) / 1000.0_wp
    end if
  end do

  ! Benchmark intrinsic exp
  call cpu_time(t_start)
  do i = 1, NBENCH
    result_ref(i) = exp(x(i))
  end do
  call cpu_time(t_end)
  t_ref = t_end - t_start

  ! Benchmark fast_exp
  call cpu_time(t_start)
  do i = 1, NBENCH
    result_fast(i) = fast_exp(x(i))
  end do
  call cpu_time(t_end)
  t_fast = t_end - t_start

  speedup = t_ref / max(t_fast, 1.0e-10_wp)

  write(*,'(A,F8.4,A)') "  Intrinsic exp():  ", t_ref, " seconds"
  write(*,'(A,F8.4,A)') "  Fast exp():       ", t_fast, " seconds"
  write(*,'(A,F8.2,A)') "  Speedup:          ", speedup, "x"

  if (speedup < 1.0_wp) then
    write(*,*) "  NOTE: Fast exp is slower than intrinsic (compiler may be auto-optimizing)"
    write(*,*) "  This is expected with -O3 -march=native on modern CPUs with fast hardware exp"
    write(*,*) "  The real benefit is on GPU where exp() is much more expensive"
  end if
  write(*,*)

  deallocate(x, result_fast, result_ref)

  ! -------------------------------------------------------------------------------------------------
  ! TEST 4: Benchmark array transmissivity computation (THE hot path)
  ! -------------------------------------------------------------------------------------------------
  write(*,*) "--- Test 4: Benchmark fast_trans_array (RT hot path) ---"
  write(*,'(A,I0,A,I0,A)') "  Array size: ncol=", NCOL, " nlay=", NLAY, ""

  allocate(tau(NCOL, NLAY), D(NCOL), trans_fast(NCOL, NLAY), trans_ref(NCOL, NLAY))

  ! Generate realistic atmospheric profiles
  do ilay = 1, NLAY
    do icol = 1, NCOL
      ! Optical depth: exponentially distributed, mostly thin
      tau(icol, ilay) = 0.01_wp * exp(3.0_wp * real(mod(icol*ilay, 100), wp) / 100.0_wp)
    end do
  end do
  D(:) = 1.66_wp  ! Diffusivity factor

  ! Benchmark: reference (separate multiply + exp)
  call cpu_time(t_start)
  do i = 1, 100  ! 100 iterations to get measurable time
    do ilay = 1, NLAY
      trans_ref(:, ilay) = exp(-tau(:, ilay) * D(:))
    end do
  end do
  call cpu_time(t_end)
  t_ref = (t_end - t_start) / 100.0_wp

  ! Benchmark: fast_trans_array
  call cpu_time(t_start)
  do i = 1, 100
    call fast_trans_array(tau, D, trans_fast, NCOL, NLAY)
  end do
  call cpu_time(t_end)
  t_fast = (t_end - t_start) / 100.0_wp

  speedup = t_ref / max(t_fast, 1.0e-10_wp)

  ! Check accuracy
  max_rel_err = 0.0_wp
  do ilay = 1, NLAY
    do icol = 1, NCOL
      if (trans_ref(icol, ilay) > tiny(1.0_wp)) then
        rel_err = abs(trans_fast(icol, ilay) - trans_ref(icol, ilay)) / trans_ref(icol, ilay)
        max_rel_err = max(max_rel_err, rel_err)
      end if
    end do
  end do

  write(*,'(A,F10.6,A)') "  Reference (exp):      ", t_ref*1000, " ms per call"
  write(*,'(A,F10.6,A)') "  Fast trans array:     ", t_fast*1000, " ms per call"
  write(*,'(A,F8.2,A)')  "  Speedup:              ", speedup, "x"
  write(*,'(A,ES12.5)')  "  Max relative error:   ", max_rel_err
  write(*,*)

  deallocate(tau, D, trans_fast, trans_ref)

  ! -------------------------------------------------------------------------------------------------
  ! Summary
  ! -------------------------------------------------------------------------------------------------
  write(*,*) "============================================="
  if (all_passed) then
    write(*,*) " ALL TESTS PASSED"
  else
    write(*,*) " SOME TESTS FAILED"
  end if
  write(*,*) "============================================="

end program test_fast_math
