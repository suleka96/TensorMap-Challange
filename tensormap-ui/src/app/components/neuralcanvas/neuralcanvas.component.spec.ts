import { async, ComponentFixture, TestBed } from '@angular/core/testing';

import { NeuralcanvasComponent } from './neuralcanvas.component';

describe('NeuralcanvasComponent', () => {
  let component: NeuralcanvasComponent;
  let fixture: ComponentFixture<NeuralcanvasComponent>;

  beforeEach(async(() => {
    TestBed.configureTestingModule({
      declarations: [ NeuralcanvasComponent ]
    })
    .compileComponents();
  }));

  beforeEach(() => {
    fixture = TestBed.createComponent(NeuralcanvasComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
