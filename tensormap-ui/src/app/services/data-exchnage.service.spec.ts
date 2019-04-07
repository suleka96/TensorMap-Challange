import { TestBed } from '@angular/core/testing';

import { DataExchnageService } from './data-exchnage.service';

describe('DataExchnageService', () => {
  beforeEach(() => TestBed.configureTestingModule({}));

  it('should be created', () => {
    const service: DataExchnageService = TestBed.get(DataExchnageService);
    expect(service).toBeTruthy();
  });
});
