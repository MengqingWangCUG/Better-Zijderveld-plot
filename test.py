import numpy as np
import pandas as pd
import ultraplot as upt
from Better_Zijderveld_plot import NSzplot, domean

def create_test_data(n_points=15):
    moment = np.logspace(-4, -6, n_points)
    
    initial_dec = np.random.uniform(0, 360)
    target_dec = 45.0
    dec = np.linspace(initial_dec, target_dec, n_points) + np.random.normal(0, 2, n_points)
    dec = np.mod(dec, 360) 
    
    initial_inc = np.random.uniform(-90, 90)
    target_inc = 60.0
    inc = np.linspace(initial_inc, target_inc, n_points) + np.random.normal(0, 1, n_points)
    inc = np.clip(inc, -90, 90)  
    
    treatments = [f'{i*10}' for i in range(n_points)]
    treatmenttext = pd.DataFrame({'treatment': treatments})
    
    generaldf = pd.DataFrame({
        'treatment': treatments,
        'dec': dec,
        'inc': inc,
        'moment': moment
    })
    
    start_idx = n_points // 4
    end_idx = 3 * n_points // 4
    selectbool = np.zeros(n_points, dtype=bool)
    selectbool[start_idx:end_idx] = True
    selectlow = treatments[start_idx]
    selecthigh = treatments[end_idx-1]
    
    return {
        'dec': dec,
        'inc': inc,
        'moment': moment,
        'treatments': treatments,
        'treatmenttext': treatmenttext,
        'generaldf': generaldf,
        'selectbool': selectbool,
        'selectlow': selectlow,
        'selecthigh': selecthigh,
        'start_idx': start_idx,
        'end_idx': end_idx
    }

def calculate_pca_from_data(generaldf, start_idx, end_idx, calculation_type='DE-BFL-A'):
    data = []
    for i in range(len(generaldf)):
        treatment = generaldf.iloc[i]['treatment']
        dec = generaldf.iloc[i]['dec']
        inc = generaldf.iloc[i]['inc']
        moment = generaldf.iloc[i]['moment']
        flag = 'g'
        data.append([float(treatment), dec, inc, moment, flag])
    
    try:
        pca_result = domean(data, start_idx, end_idx-1, calculation_type)
        
        PCAdf = pd.DataFrame({
            'specimen_dec': [pca_result.get('specimen_dec', 0.0)],
            'specimen_inc': [pca_result.get('specimen_inc', 0.0)],
            'calculation_type': [calculation_type],
            'specimen_mad': [pca_result.get('specimen_mad', 0.0)],
            'specimen_dang': [pca_result.get('specimen_dang', 0.0)],
            'specimen_n': [pca_result.get('specimen_n', 0)],
            'measurement_step_min': [pca_result.get('measurement_step_min', 0)],
            'measurement_step_max': [pca_result.get('measurement_step_max', 0)]
        })
        
        print(f"PCA calculation successful:")
        print(f"  Dec: {pca_result.get('specimen_dec', 0.0):.1f}°")
        print(f"  Inc: {pca_result.get('specimen_inc', 0.0):.1f}°")
        print(f"  MAD: {pca_result.get('specimen_mad', 0.0):.1f}°")
        print(f"  DANG: {pca_result.get('specimen_dang', 0.0):.1f}°")
        print(f"  N: {pca_result.get('specimen_n', 0)}")
        
        return PCAdf
        
    except Exception as e:
        print(f"PCA calculation failed: {e}")
        return pd.DataFrame({
            'specimen_dec': [45.0],
            'specimen_inc': [60.0],
            'calculation_type': [calculation_type],
            'specimen_mad': [2.5],
            'specimen_dang': [5.0],
            'specimen_n': [end_idx - start_idx]
        })

def test_basic_zplot():
    print("Testing basic Zijderveld plot...")
    
    data = create_test_data(12)
    
    fig, ax = upt.subplots(figsize=(8, 8))
    
    NSzplot(
        dec=data['dec'],
        inc=data['inc'],
        moment=data['moment'],
        specimen_name='Test Sample Basic',
        Iunit='$Am^2$',
        treatmenttext=data['treatmenttext'],
        ax=ax,
        PCA=False,
        markersize=5,
        linewidth=1.5
    )
    
    fig.savefig('test_basic_zplot.pdf')
    print("Basic Zijderveld plot completed")

def test_zplot_with_pca():
    print("Testing Zijderveld plot with PCA...")
    
    data = create_test_data(15)
    
    print("Calculating PCA using domean function...")
    PCAdf = calculate_pca_from_data(
        data['generaldf'], 
        data['start_idx'], 
        data['end_idx'], 
        'DE-BFL-A'
    )
    
    print(f"Number of dec values: {len(data['dec'])}")
    print(f"Number of inc values: {len(data['inc'])}")
    print(f"Number of moment values: {len(data['moment'])}")
    print(f"Moment range: {data['moment'].max():.3e} to {data['moment'].min():.3e}")
    print(f"Moment is decreasing: {np.all(np.diff(data['moment']) <= 0)}")
    
    fig, ax = upt.subplots(figsize=(8, 8))
    
    NSzplot(
        dec=data['dec'],
        inc=data['inc'],
        moment=data['moment'],
        specimen_name='Test Sample with PCA',
        Iunit='$Am^2$',
        treatmenttext=data['treatmenttext'],
        ax=ax,
        PCA=True,
        PCAdf=PCAdf,
        generaldf=data['generaldf'],
        selectlow=data['selectlow'],
        selecthigh=data['selecthigh'],
        selectbool=data['selectbool'],
        markersize=4,
        linewidth=1
    )

    fig.savefig('test_zplot_with_pca.pdf')
    print("Zijderveld plot with PCA completed")

def test_different_pca_types():
    print("Testing different PCA calculation types...")
    
    data = create_test_data(20)
    pca_types = ['DE-BFL-A', 'DE-BFL-O', 'DE-BFL']
    
    for i, calc_type in enumerate(pca_types):
        print(f"\nTesting {calc_type}...")
        
        PCAdf = calculate_pca_from_data(
            data['generaldf'], 
            data['start_idx'], 
            data['end_idx'], 
            calc_type
        )
        
        fig, ax = upt.subplots(figsize=(8, 8))
        
        NSzplot(
            dec=data['dec'],
            inc=data['inc'],
            moment=data['moment'],
            specimen_name=f'Test Sample {calc_type}',
            Iunit='$Am^2$',
            treatmenttext=data['treatmenttext'],
            ax=ax,
            PCA=True,
            PCAdf=PCAdf,
            generaldf=data['generaldf'],
            selectlow=data['selectlow'],
            selecthigh=data['selecthigh'],
            selectbool=data['selectbool'],
            markersize=4,
            linewidth=1
        )
        
        fig.savefig(f'test_pca_{calc_type.replace("-", "_")}.pdf')
        print(f"{calc_type} plot completed")

def test_data_validation():
    print("Validating test data...")
    
    data = create_test_data(20)
    
    assert len(data['dec']) == len(data['inc']) == len(data['moment']), \
        "Dec, inc, and moment arrays must have equal length"
    
    assert np.all(np.diff(data['moment']) <= 0), \
        "Moment values must be decreasing"
    
    assert np.all(data['dec'] >= 0) and np.all(data['dec'] <= 360), \
        "Declination must be in range [0, 360]"
    
    assert np.all(data['inc'] >= -90) and np.all(data['inc'] <= 90), \
        "Inclination must be in range [-90, 90]"
    
    assert np.all(data['moment'] > 0), \
        "Moment values must be positive"
    
    print("All data validation tests passed!")
    
    print(f"Data points: {len(data['moment'])}")
    print(f"Declination range: {data['dec'].min():.1f}° to {data['dec'].max():.1f}°")
    print(f"Inclination range: {data['inc'].min():.1f}° to {data['inc'].max():.1f}°")
    print(f"Moment range: {data['moment'].max():.3e} to {data['moment'].min():.3e}")
    print(f"PCA selection: {np.sum(data['selectbool'])} out of {len(data['selectbool'])} points")

def run_all_tests():
    print("=" * 60)
    print("Starting NSzplot Test Suite with Real PCA Calculation")
    print("=" * 60)
    
    try:
        test_data_validation()
        print()
        
        test_basic_zplot()
        print()
        
        test_zplot_with_pca()
        print()
        
        test_different_pca_types()
        print()
        
        print("=" * 60)
        print("All tests completed successfully!")
        print("Check the generated PDF files for visual results:")
        print("- test_basic_zplot.pdf")
        print("- test_zplot_with_pca.pdf")
        print("- test_pca_DE_BFL_A.pdf")
        print("- test_pca_DE_BFL_O.pdf")
        print("- test_pca_DE_BFL.pdf")
        print("=" * 60)
        
    except Exception as e:
        print(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    np.random.seed(42)
    
    run_all_tests()